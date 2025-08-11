"""Package entrypoints for getcontacts scripts."""

import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

GETCONTACTS_SCRIPTS_DIR = Path(__file__).parent.parent.parent / "external" / "getcontacts"


def _create_script_runner(script_name: str) -> Callable[[], None]:
    def runner() -> None:
        import multiprocessing as mp
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
        print(f"[getcontacts] start_method={mp.get_start_method()}", file=sys.stderr)
        script_path = GETCONTACTS_SCRIPTS_DIR / f"{script_name}.py"
        subprocess.run([sys.executable, str(script_path), *sys.argv[1:]], check=False)

    return runner


get_contact_bridges = _create_script_runner("get_contact_bridges")
get_contact_fingerprints = _create_script_runner("get_contact_fingerprints")
get_contact_flare = _create_script_runner("get_contact_flare")
get_contact_frequencies = _create_script_runner("get_contact_frequencies")
get_contact_singleframe = _create_script_runner("get_contact_singleframe")
get_contact_ticc = _create_script_runner("get_contact_ticc")
get_contact_trace = _create_script_runner("get_contact_trace")
get_dynamic_contacts = _create_script_runner("get_dynamic_contacts")
get_resilabels = _create_script_runner("get_resilabels")
get_static_contacts = _create_script_runner("get_static_contacts")
