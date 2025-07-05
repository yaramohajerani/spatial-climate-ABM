"""Run prototype climate-economy ABM for 10 timesteps and persist results."""

import argparse
from model import EconomyModel
# Runner now expects one or more --rp-file arguments in the form "<RP>:<TYPE>:<path>"

def _parse():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--rp-file",
        action="append",
        metavar="RP:TYPE:PATH",
        help="Add a GeoTIFF file. Format: <RP>:<HAZARD_TYPE>:<path>. Example: --rp-file 100:FL:rp100.tif",
        required=True,
    )
    p.add_argument("--viz", action="store_true", help="Launch interactive Solara dashboard instead of headless run")
    return p.parse_args()

def main() -> None:  # noqa: D401
    args = _parse()

    # First, parse the RP files into a list irrespective of --viz so we can
    # pass them on to a potential Solara dashboard.
    events: list[tuple[int, str, str]] = []
    if args.rp_file:
        for item in args.rp_file:
            try:
                rp_str, type_str, path_str = item.split(":", 2)
                events.append((int(rp_str), type_str, path_str))
            except ValueError as exc:  # noqa: BLE001
                raise SystemExit(
                    f"Invalid --rp-file format: {item}. Expected <RP>:<TYPE>:<path>."
                ) from exc

    # If visualization requested, delegate to Solara which hosts the dashboard
    if args.viz:
        import subprocess, sys, os

        env = os.environ.copy()
        # Pass hazard events to the dashboard so it can build the same model
        env["ABM_HAZARD_EVENTS"] = ";".join(f"{rp}:{t}:{p}" for rp, t, p in events)

        cmd = [sys.executable, "-m", "solara", "run", "visualization.py"]
        subprocess.run(cmd, env=env, check=False)
        return

    # Headless mode: run the simulation directly
    model = EconomyModel(
        num_households=100,
        num_firms=20,
        shock_step=5,
        hazard_events=events,
    )

    for _ in range(10):  # simulate 10 years
        model.step()

    model.save_results("simulation_results")
    print("Simulation complete. Results stored in simulation_results.csv")


if __name__ == "__main__":
    main() 