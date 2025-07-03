"""Run prototype climate-economy ABM for 10 timesteps and persist results."""

import argparse
from model import EconomyModel
# Runner now expects one or more --rp-file arguments in the form "<RP>:<path>"

def _parse():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--rp-file",
        action="append",
        metavar="RP:PATH",
        help="Add a GeoTIFF file defining flood depth for the given return period. Example: --rp-file 10:rp10.tif",
        required=True,
    )
    p.add_argument("--viz", action="store_true", help="Launch interactive Solara dashboard instead of headless run")
    return p.parse_args()

def main() -> None:  # noqa: D401
    args = _parse()

    # If visualization requested, delegate to Solara which hosts the dashboard
    if args.viz:
        import subprocess, sys, os

        env = os.environ.copy()
        # No need to set hazard-related environment variables as they are handled by the model

        cmd = [sys.executable, "-m", "solara", "run", "visualization.py"]
        subprocess.run(cmd, env=env, check=False)
        return

    # Parse RP files into {int: str}
    rp_map = {}
    for item in args.rp_file:
        try:
            rp_str, path_str = item.split(":", 1)
            rp_map[int(rp_str)] = path_str
        except ValueError as exc:  # noqa: BLE001
            raise SystemExit(f"Invalid --rp-file format: {item}. Expected <RP>:<path>.") from exc

    model = EconomyModel(
        # Grid dimensions inferred automatically from the raster set
        num_households=100,
        num_firms=20,
        shock_step=5,
        hazard_rp_files=rp_map,
    )

    for _ in range(10):  # simulate 10 years
        model.step()

    model.save_results("simulation_results")
    print("Simulation complete. Results stored in simulation_results.csv")


if __name__ == "__main__":
    main() 