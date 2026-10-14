"""Headless screenshot runner for the development Coral experiment.

Save this file beside ``experiments/coral_dev/run.py`` and run, for example:
    python experiments/coral_dev/headless_coral_dev.py --steps 300 --env patches
"""

import argparse
import csv
import json
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import taichi as ti
from PIL import Image


parser = argparse.ArgumentParser(description="Headless Coral Dev runner with PNG output")
parser.add_argument("--steps", type=int, required=True)
parser.add_argument("--shape", type=int, default=400)
parser.add_argument("--backend", choices=["cpu", "metal", "cuda", "vulkan"], default="cpu")
parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--env", default="flat",
                    choices=["flat", "hole", "ring", "stripes", "corners", "patches", "oases"])
parser.add_argument("--env-param", type=float, default=None)
parser.add_argument("--infra-decay", type=float, default=0.0)
parser.add_argument("--defense-coeff", type=float, default=0.0)
parser.add_argument("--radiate-interval", type=int, default=50)
parser.add_argument("--cleanup-interval", type=int, default=100)
parser.add_argument("--emergency-pop-cap", type=int, default=1000)
parser.add_argument("--checkpoint-interval", type=int, default=1000)
parser.add_argument("--log-interval", type=int, default=10)
parser.add_argument("--screenshot-interval", type=int, default=1,
                    help="Save one PNG every N simulation steps")
parser.add_argument("--max-screenshots", type=int, default=2000)
parser.add_argument("--resume-from", type=str, default=None)
parser.add_argument("--output-dir", type=Path, default=None)
args = parser.parse_args()

if args.steps < 1:
    parser.error("--steps must be positive")
if args.screenshot_interval < 1:
    parser.error("--screenshot-interval must be positive")

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

backend_map = {"cpu": ti.cpu, "metal": ti.metal, "cuda": ti.cuda, "vulkan": ti.vulkan}
ti.init(arch=backend_map[args.backend])
DEVICE = torch.device(args.device)

# This file lives in experiments/coral_dev, so these are the same imports as run.py.
from coralai.evolver import apply_weights_and_biases
from experiment import EXPERIMENT as exp
from evolution import apply_radiation_mutation

exp.infra_decay = args.infra_decay
exp.defense_coeff = args.defense_coeff


def sync():
    if args.device == "mps":
        torch.mps.synchronize()
    elif args.device == "cuda":
        torch.cuda.synchronize()


def scalar_channel(substrate, name):
    """Return one (W, H) channel even if ``name`` names a struct/vector."""
    index = substrate.windex[name]
    if isinstance(index, (list, tuple)):
        index = index[0]
    values = substrate.mem[0, index].detach().cpu()
    if values.ndim > 2:
        values = values.reshape(-1, *values.shape[-2:]).mean(dim=0)
    return values.numpy()


def make_frame(substrate):
    """Match the normal visualizer's state view: energy / infra / genome."""
    image = np.zeros((substrate.h, substrate.w, 3), dtype=np.float32)
    for color, name in enumerate(("energy", "infra", "genome")):
        channel = scalar_channel(substrate, name)
        lo, hi = channel.min(), channel.max()
        if hi > lo:
            channel = (channel - lo) / (hi - lo)
        else:
            channel = np.zeros_like(channel)
        image[..., color] = channel.T
    return np.clip(image * 255, 0, 255).astype(np.uint8)


def main():
    shape = (args.shape, args.shape)
    substrate = exp.make_substrate(shape, DEVICE)
    env = exp.make_env(args.env, args.env_param)
    env.seed(substrate)
    evolver = exp.make_evolver(substrate)

    if args.resume_from:
        evolver.load_checkpoint(args.resume_from)
        print(f"[resume] loaded {args.resume_from} (step {evolver.timestep})")

    start_step = evolver.timestep
    run_dir = args.output_dir or (
        Path(__file__).resolve().parents[2] / "runs" /
        f"coral_dev_headless_{datetime.now():%Y%m%d_%H%M%S}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    screenshots_dir = run_dir / "screenshots"
    screenshots_dir.mkdir(exist_ok=True)

    with (run_dir / "meta.json").open("w") as file:
        json.dump({
            "experiment": exp.name, "shape": list(shape), "seed": args.seed,
            "env": env.to_dict(), "env_persist": env.persist,
            "start_step": start_step, "resumed_from": args.resume_from,
            "infra_decay": args.infra_decay, "defense_coeff": args.defense_coeff,
        }, file, indent=2)
    torch.save(substrate.mem.cpu(), run_dir / "initial_state.pt")

    log_file = (run_dir / "step_log.csv").open("w", newline="")
    log = csv.writer(log_file)
    log.writerow(["step", "total_energy", "total_infra", "energy_pct", "n_alive", "n_genomes"])

    screenshot_count = 0
    last_captured_step = None

    def capture(step):
        nonlocal screenshot_count, last_captured_step
        if screenshot_count < args.max_screenshots:
            path = screenshots_dir / f"step_{step:07d}.png"
            Image.fromarray(make_frame(substrate)).save(path)
            screenshot_count += 1
            last_captured_step = step

    capture(start_step)
    inds = substrate.ti_indices[None]
    end_step = start_step + args.steps
    started = time.perf_counter()
    try:
        for step in range(start_step, end_step):
            # Same step order as coral_dev/run.py.
            weights, biases = evolver.get_combined_weights()
            out_mem = evolver._get_scratch("out_mem", substrate.mem[0, evolver.act_chinds])
            apply_weights_and_biases(
                substrate.mem, out_mem, evolver.sense_chinds, weights, biases,
                evolver.dir_kernel, evolver.dir_order, substrate.ti_indices,
                evolver._get_key_to_local(),
            )
            substrate.mem[0, evolver.act_chinds] = out_mem
            exp.run_physics(substrate, evolver)
            exp.run_evolution(substrate, evolver, step)
            if env.persist:
                env.step(substrate)
            if step > 0 and step % args.radiate_interval == 0:
                apply_radiation_mutation(evolver, 5)
            if step > 0 and step % args.cleanup_interval == 0:
                evolver.remove_extinct_genomes()
            if (args.emergency_pop_cap > 0
                    and len(evolver.genomes) > args.emergency_pop_cap
                    and step - evolver.time_last_cull > 50):
                evolver.reduce_population_to_threshold(args.emergency_pop_cap)
            evolver.timestep = step + 1

            if evolver.timestep % args.screenshot_interval == 0:
                capture(evolver.timestep)
            if step % args.log_interval == 0:
                total_energy = float(substrate.mem[0, inds.energy].sum())
                total_infra = float(substrate.mem[0, inds.infra].sum())
                alive = int((substrate.mem[0, inds.genome] >= 0).sum())
                log.writerow([step, total_energy, total_infra,
                              100 * total_energy / (total_energy + total_infra + 1e-8),
                              alive, len(evolver.genomes)])
                log_file.flush()
            if (args.checkpoint_interval > 0 and step > 0
                    and step % args.checkpoint_interval == 0):
                evolver.save_checkpoint(run_dir, step)

        # Always include the final state, unless it was already sampled.
        if last_captured_step != evolver.timestep:
            capture(evolver.timestep)
    finally:
        log_file.close()

    elapsed = time.perf_counter() - started
    print(f"{args.steps} steps in {elapsed:.2f}s ({args.steps / elapsed:.1f} FPS)")
    print(f"Screenshots: {screenshots_dir} ({screenshot_count} PNGs)")
    print(f"Run data: {run_dir}")


if __name__ == "__main__":
    main()
