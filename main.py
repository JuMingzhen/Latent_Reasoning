import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict
from run import run_train, run_evaluate, run_predict

def load_config(path: Path) -> Dict[str, Any]:
    """Load config from JSON or YAML (if PyYAML is installed)."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))

    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "PyYAML is required for YAML configs. "
                "Install with `pip install pyyaml` or use JSON instead."
            ) from exc
        return yaml.safe_load(path.read_text(encoding="utf-8"))

    raise ValueError(f"Unsupported config format: {suffix}")


def run_task(task: str, cfg: Dict[str, Any]) -> None:
    """Dispatch task name to a simple demo implementation."""
    handlers = {
        "train": run_train,
        "evaluate": run_evaluate,
        "predict": run_predict,
    }
    try:
        handler = handlers[task]
    except KeyError:
        raise ValueError(f"Unknown task '{task}'. Supported: {list(handlers)}")

    handler(cfg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Demo: start tasks using parameters from a config file."
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=Path("config/config.example.yaml"),
        help="Path to config file (YAML or JSON).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config
    if not config_path.exists() and config_path.parent == Path("."):
        candidate = Path("config") / config_path.name
        if candidate.exists():
            config_path = candidate
    cfg = load_config(config_path)
    task = cfg.get("task")
    if not task:
        raise ValueError("Config must include a top-level 'task' field.")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logging.info("Loaded config from %s", args.config)
    run_task(task, cfg)


if __name__ == "__main__":
    main()

