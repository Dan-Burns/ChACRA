# -*- coding: utf-8 -*-
"""
Run configuration manager for ChACRA HREMD runs.

Reads and writes ``chacra_run.json`` in the project root directory so that
simulation parameters are persisted across restarts without requiring the user
to re-supply every CLI flag.
"""

import json
import os
import psutil


def get_resources() -> dict:
    """
    Return a dict of available CPU and RAM resources on this machine.

    Returns
    -------
    dict with keys:
        num_cores, num_threads, total_ram_gb, available_ram_gb, available_ram_mb
    """
    return {
        "num_cores": psutil.cpu_count(logical=False),   # physical cores
        "num_threads": psutil.cpu_count(logical=True),  # includes hyperthreads
        "total_ram_gb": psutil.virtual_memory().total / 1e9,
        "available_ram_gb": psutil.virtual_memory().available / 1e9,
        "available_ram_mb": psutil.virtual_memory().available / 1e6,
    }


class RunConfig:
    """
    Configuration manager for ChACRA HREMD runs.

    Reads and writes ``chacra_run.json`` in the project root directory.
    On the first run this file is created with all resolved parameters
    (including the full temperature list).  On subsequent runs it is loaded
    automatically so the user does not need to re-supply every CLI flag.
    CLI arguments always take precedence over values stored in the JSON.

    Parameters
    ----------
    config_file : str or None
        Path to an existing ``chacra_run.json``.  If *None* an empty config
        is created using built-in defaults.

    Attributes
    ----------
    defaults : dict
        Hard-coded default values for every parameter.
    config : dict
        The resolved configuration.  Starts from ``defaults`` and is
        overridden by any values loaded from *config_file*.
    """

    #: Default path written/read in the project root directory.
    DEFAULT_PATH: str = "chacra_run.json"

    defaults: dict = {
        "n_jobs": None,
        "n_cycles": 1000,
        "n_systems": None,
        "min_temp": 290.0,
        "max_temp": 450.0,
        "temps": None,          # populated after first run
        "structure_file": None,
        "system_file": None,
        "steps_per_cycle": 1000,
        "save_interval": 10,
        "checkpoint_interval": 500,
        "warmup_steps": 0,
        "lambda_selection": "protein",
        "output_selection": "protein",
        "timestep": 2,
        "current_run": 0,
        "oversubscribe": 1,
    }

    def __init__(self, config_file: str | None = None):
        self.config_file = config_file
        # Start from defaults, then overlay file contents
        self.config: dict = dict(self.defaults)

        if config_file is not None:
            if os.path.exists(config_file):
                self._load(config_file)
            else:
                print(f"[RunConfig] Config file not found: {config_file}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write(self, path: str | None = None) -> None:
        """
        Serialise ``self.config`` to JSON.

        Parameters
        ----------
        path : str or None
            Destination path.  Defaults to :attr:`DEFAULT_PATH`
            (``chacra_run.json`` in the current working directory).
        """
        dest = path or self.DEFAULT_PATH
        with open(dest, "w") as fh:
            json.dump(self.config, fh, indent=2)
        print(f"[RunConfig] Config written to {dest}")

    def update(self, **kwargs) -> None:
        """
        Update individual config keys, ignoring keys whose value is *None*
        (so CLI arg defaults don't silently erase stored values).

        Parameters
        ----------
        **kwargs
            Key-value pairs to merge into ``self.config``.
        """
        for key, value in kwargs.items():
            if value is not None:
                self.config[key] = value

    def apply_to_namespace(self, namespace) -> None:
        """
        Back-fill an ``argparse.Namespace`` with config values for any
        attribute that is currently *None*.  This lets CLI-supplied arguments
        always win while filling gaps from the JSON.

        Parameters
        ----------
        namespace : argparse.Namespace
        """
        for key, value in self.config.items():
            if getattr(namespace, key, None) is None and value is not None:
                setattr(namespace, key, value)

    def get(self, key: str, default=None):
        """Return a config value, falling back to *default*."""
        return self.config.get(key, default)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load(self, path: str) -> None:
        """Load JSON from *path* and overlay onto ``self.config``."""
        try:
            with open(path, "r") as fh:
                data = json.load(fh)
            self.config.update(data)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[RunConfig] Could not read config file '{path}': {exc}")
