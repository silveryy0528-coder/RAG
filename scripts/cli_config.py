"""Shared YAML config loader for RAG CLI scripts.

Each script declares its own ``ALLOWED_CONFIG_KEYS``, ``BOOL_CONFIG_KEYS``,
and ``PATH_CONFIG_KEYS`` and passes them to :func:`load_config`.  The loader
validates the file against those sets so that scripts cannot silently accept
unknown options from each other's configs.

Usage pattern (in any CLI ``parse_args``)::

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path, default=None)
    pre_args, _ = pre_parser.parse_known_args(argv)
    defaults = {}
    if pre_args.config is not None:
        defaults = load_config(
            pre_args.config.expanduser(),
            allowed=ALLOWED_CONFIG_KEYS,
            bool_keys=BOOL_CONFIG_KEYS,
            path_keys=PATH_CONFIG_KEYS,
        )
    parser.set_defaults(**defaults)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_config(
    config_path: Path,
    *,
    allowed: set[str],
    bool_keys: set[str],
    path_keys: set[str],
) -> dict[str, Any]:
    """Load and validate a YAML config file into a flat defaults dictionary.

    Parameters
    ----------
    config_path:
        Path to the YAML file.
    allowed:
        Full set of recognised config keys for the calling script.
    bool_keys:
        Subset of *allowed* keys that must be Python booleans.
    path_keys:
        Subset of *allowed* keys whose non-null values are coerced to
        :class:`pathlib.Path`.

    Returns
    -------
    dict[str, Any]
        Flat mapping suitable for passing to ``parser.set_defaults(**defaults)``.

    Raises
    ------
    FileNotFoundError
        If *config_path* does not exist.
    RuntimeError
        If PyYAML is not installed.
    ValueError
        If the YAML contains unknown keys or values with the wrong type.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        import yaml  # type: ignore[import]
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "YAML config support requires PyYAML. "
            "Install it with `pip install pyyaml`."
        ) from exc

    with config_path.open("r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh)

    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError("Config file must contain a YAML mapping at the top level.")

    unknown = sorted(set(loaded) - allowed)
    if unknown:
        raise ValueError(f"Unknown config keys in {config_path}: {', '.join(unknown)}")

    result: dict[str, Any] = {}
    for key, value in loaded.items():
        if key in bool_keys and not isinstance(value, bool):
            raise ValueError(f"Config key '{key}' must be a boolean (true/false).")
        if key == "k" and not isinstance(value, int):
            raise ValueError("Config key 'k' must be an integer.")
        if key == "temperature" and not isinstance(value, (int, float)):
            raise ValueError("Config key 'temperature' must be a number.")
        if key in path_keys and value is not None:
            result[key] = Path(value)
            continue
        result[key] = value
    return result
