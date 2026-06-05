"""
generate_demo_data.py
---------------------
Generates data for the PINN interactive demo at 32×32 FDM grid.

Output per (problem, model_type):
    data/{problem}/{model_type}/32x32/
        predictions.json        {"data": [[...]]}   PINN u_pinn
        true_solution.json      {"data": [[...]]}   exact u
        error_true.json         {"data": [[...]]}   |u_exact − u_pinn|  (true pointwise error)
        error_estimate.json     {"data": [[...]]}   |e_res| from FDM residual integration
        error_diff.json         {"data": [[...]]}   error_estimate − error_true  (signed)
        meta.json               stats + axes + config

Model types
-----------
  well_trained_model          10k iters, 10k colloc, seed 42
  randomly_initialized_model  0 iters, raw weights, seed 42

Usage
-----
    python generate_demo_data.py
    python generate_demo_data.py --output-dir demo/data
    python generate_demo_data.py --dry-run
    python generate_demo_data.py --no-skip
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import deepxde as dde

from pinn_error.core.pinn import PINNConfig, PINNTrainer
from pinn_error.utils.experiment_factory import ExperimentFactory, get_problem

# ── device + dtype ─────────────────────────────────────────────────────────────
_MPS = torch.backends.mps.is_built() and torch.backends.mps.is_available()
if _MPS:
    print("MPS detected — forcing CPU + float64 for numerical stability.")
torch.set_default_device("cpu")
dde.config.set_default_float("float64")

# ── fixed settings ─────────────────────────────────────────────────────────────
SEED   = 42
LAYERS = [2, 20, 20, 20, 1]
GRID   = 32

MODEL_TYPES = {
    "well_trained_model": {
        "label":        "Well-Trained Model",
        "n_iterations": 10_000,
        "num_domain":   10_000,
        "trained":      True,
    },
    "randomly_initialized_model": {
        "label":       "Randomly Initialised Model",
        "n_iterations": 0,
        "num_domain":   0,
        "trained":      False,
    },
}

PROBLEMS = {
    "heat": {
        "label":       "1-D Heat Equation",
        "description": "u_t = α u_xx · sinusoidal IC, zero Dirichlet BCs",
        "problem_kwargs": {
            "x_min": 0.0, "x_max": 1.0, "t_max": 1.0,
            "diffusivity": 0.05, "frequency": 2,
        },
        "time_dependent": True,
        "is_2d": False,
        "x_label": "x", "y_label": "t",
    },
    "wave": {
        "label":       "1-D Wave Equation",
        "description": "u_tt = c² u_xx · sinusoidal IC, zero Dirichlet BCs",
        "problem_kwargs": {
            "x_min": 0.0, "x_max": 1.0, "t_max": 1.0,
            "propagation_speed": 0.5, "frequency": 1,
        },
        "time_dependent": True,
        "is_2d": False,
        "x_label": "x", "y_label": "t",
    },
    "drift_diffusion": {
        "label":       "1-D Drift-Diffusion",
        "description": "u_t + β u_x = D u_xx · sinusoidal IC",
        "problem_kwargs": {
            "x_min": 0.0, "x_max": 1.0, "t_max": 0.5,
            "diffusivity": 0.05, "velocity_x": 2.0,
            "frequency": 2.0, "phase_shift": 0.0,
            "initial_concentration": 1.0,
        },
        "time_dependent": True,
        "is_2d": False,
        "x_label": "x", "y_label": "t",
    },
    "poisson_2d": {
        "label":       "2-D Poisson Equation",
        "description": "-(u_xx + u_yy) = f(x,y) · zero Dirichlet BCs",
        "problem_kwargs": {
            "x_min": 0.0, "x_max": 1.0,
            "y_min": 0.0, "y_max": 1.0,
        },
        "time_dependent": False,
        "is_2d": True,
        "x_label": "x", "y_label": "y",
    },
}

# ── helpers ────────────────────────────────────────────────────────────────────

def _write(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, separators=(",", ":"))


def _stats(e_true: np.ndarray, u_exact: np.ndarray, e_est: np.ndarray) -> dict:
    return {
        "max_error_true":     float(np.max(np.abs(e_true))),
        "mean_error_true":    float(np.mean(np.abs(e_true))),
        "l2_relative":        float(np.sqrt(np.mean(e_true**2)) / (np.sqrt(np.mean(u_exact**2)) + 1e-12)),
        "max_error_estimate": float(np.max(np.abs(e_est))),
        "mean_error_estimate":float(np.mean(np.abs(e_est))),
    }


def _run_well_trained(prob_key, prob_cfg, nx):
    """Run ExperimentFactory and return all arrays."""
    fdm_kw = {"nx": nx, "nt": nx} if prob_cfg["time_dependent"] else {"nx": nx, "ny": nx}
    pinn_cfg = PINNConfig(
        layers=LAYERS, n_iterations=10_000, num_domain=10_000,
        seed=SEED, use_cache=True,
    )
    factory = ExperimentFactory(
        problem_name=prob_key,
        problem_kwargs=prob_cfg["problem_kwargs"],
        fdm_solver_kwargs=fdm_kw,
        pinn_config=pinn_cfg,
        verbose=False,
    )
    res = factory.run_experiment()

    x       = res["x"]
    u_pinn  = np.asarray(res["u_pinn"])
    u_exact = np.asarray(res["u_true"])
    e_res   = np.asarray(res["e_res"])   # FDM residual integration estimate

    if prob_cfg["time_dependent"]:
        t = res["t"]
        r = (len(t), len(x))
        return x, t, u_pinn.reshape(r), u_exact.reshape(r), e_res.reshape(r)
    else:
        y = res["y"]
        r = (len(y), len(x))
        return x, y, u_pinn.reshape(r), u_exact.reshape(r), e_res.reshape(r)


def _run_random(prob_key, prob_cfg, nx):
    """Predict with untrained (random-init) PINN. No e_res available — return zeros."""
    problem  = get_problem(prob_key, **prob_cfg["problem_kwargs"])
    pinn_cfg = PINNConfig(layers=LAYERS, n_iterations=0, num_domain=100,
                          seed=SEED, use_cache=False)
    trainer  = PINNTrainer(problem=problem, config=pinn_cfg)
    # intentionally skip trainer.train()

    x = np.linspace(problem.domain.x_min, problem.domain.x_max, nx)
    if prob_cfg["time_dependent"]:
        t    = np.linspace(problem.domain.t_min, problem.domain.t_max, nx)
        X, T = np.meshgrid(x, t)
        u_pinn  = trainer.predict(np.column_stack((X.ravel(), T.ravel()))).reshape(nx, nx)
        u_exact = problem.exact_solution(X, T)
        e_res   = np.zeros_like(u_pinn)   # no FDM estimate for untrained model
        return x, t, u_pinn, u_exact, e_res
    else:
        y    = np.linspace(problem.domain.y_min, problem.domain.y_max, nx)
        X, Y = np.meshgrid(x, y)
        u_pinn  = trainer.predict(np.column_stack((X.ravel(), Y.ravel()))).reshape(nx, nx)
        u_exact = problem.exact_solution(X, Y)
        e_res   = np.zeros_like(u_pinn)
        return x, y, u_pinn, u_exact, e_res


# ── main ───────────────────────────────────────────────────────────────────────

def generate(output_dir: Path, dry_run: bool = False, skip_existing: bool = True):
    nx         = GRID
    grid_label = f"{nx}x{nx}"
    total      = len(PROBLEMS) * len(MODEL_TYPES)
    run_idx    = 0

    ALL_FILES = ["predictions.json", "true_solution.json",
                 "error_true.json", "error_estimate.json",
                 "error_diff.json", "meta.json"]

    for prob_key, prob_cfg in PROBLEMS.items():
        for model_key, model_cfg in MODEL_TYPES.items():
            run_idx += 1
            base = output_dir / prob_key / model_key / grid_label

            print(f"\n[{run_idx}/{total}]  {prob_key} / {model_key} / {grid_label}")

            if dry_run:
                print(f"  → {base}/  (dry-run)")
                continue

            if skip_existing and all((base / f).exists() for f in ALL_FILES):
                print(f"  ✓ already complete, skipping")
                continue

            t0 = time.time()
            try:
                if model_cfg["trained"]:
                    x, y_axis, u_pinn, u_exact, e_res = _run_well_trained(prob_key, prob_cfg, nx)
                else:
                    x, y_axis, u_pinn, u_exact, e_res = _run_random(prob_key, prob_cfg, nx)
            except Exception as exc:
                print(f"  ERROR: {exc}")
                continue

            e_true = u_exact - u_pinn        # signed true error
            e_diff = np.abs(e_res) - np.abs(e_true)   # estimate minus true (signed)

            stats = _stats(e_true, u_exact, e_res)

            _write(base / "predictions.json",   {"data": u_pinn.tolist()})
            _write(base / "true_solution.json",  {"data": u_exact.tolist()})
            _write(base / "error_true.json",     {"data": np.abs(e_true).tolist()})
            _write(base / "error_estimate.json", {"data": np.abs(e_res).tolist()})
            _write(base / "error_diff.json",     {"data": e_diff.tolist()})
            _write(base / "meta.json", {
                "problem":          prob_key,
                "label":            prob_cfg["label"],
                "model_type":       model_key,
                "model_label":      model_cfg["label"],
                "grid":             grid_label,
                "nx":               int(nx),
                "nt_or_ny":         int(len(y_axis)),
                "x_axis":           x.tolist(),
                "y_axis":           y_axis.tolist(),
                "x_label":          prob_cfg["x_label"],
                "y_label":          prob_cfg["y_label"],
                "pinn_iterations":  model_cfg["n_iterations"],
                "pinn_collocation": model_cfg["num_domain"],
                "pinn_layers":      LAYERS,
                "seed":             SEED,
                **stats,
            })

            elapsed = time.time() - t0
            size_kb = sum((base / f).stat().st_size for f in ALL_FILES) / 1e3
            print(
                f"  ✓ {elapsed:.1f}s | {size_kb:.1f} KB | "
                f"max_err={stats['max_error_true']:.3e} | "
                f"l2={stats['l2_relative']:.3e}"
            )

    print(f"\nDone → {output_dir.resolve()}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="data")
    p.add_argument("--no-skip",  action="store_true")
    p.add_argument("--dry-run",  action="store_true")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    generate(Path(args.output_dir), dry_run=args.dry_run, skip_existing=not args.no_skip)
