import sys
import json
from datetime import datetime
from pathlib import Path

import toml

from tasks import get_trainer
from utils import *


def save_results(run_id, config, trainer, test_scores):
    """Write per-epoch history + final test scores to outputs/results/<run_id>.json."""
    results_dir = Path(__file__).parent / "outputs" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{run_id}.json"

    history = getattr(trainer, "history", [])
    # cast every numeric to plain float for clean JSON
    epochs = []
    for rec in history:
        epochs.append({k: (int(v) if k == "epoch" else float(v)) for k, v in rec.items()})

    record = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "task": config.task,
        "model": config.model,
        "dataset": config.data.dataset,
        "epochs": epochs,                                  # per-epoch train/val/test metrics
        "test": {k: float(v) for k, v in test_scores.items()},  # final test scores
    }
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)
    print("Saved results to:", out_path)
    return out_path


def main(config_path, run_id=None):
    config = toml.load(config_path)
    config = dict_to_object(config)

    run_id = run_id or get_run_id(config.DEBUG)
    trainer = get_trainer(run_id, config)

    trainer.train()
    test_scores = trainer.test()
    trainer.log_end()

    save_results(run_id, config, trainer, test_scores)

    print("Test results:", test_scores)
    print("Run ID:", run_id)


if __name__ == "__main__":
    match sys.argv:
        case [_, config_path, run_id]:
            main(config_path, run_id)
        case [_, config_path]:
            main(config_path)
        case _:
            main("configs/config.toml")
