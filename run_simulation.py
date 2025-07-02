"""Run prototype climate-economy ABM for 10 timesteps and persist results."""

import argparse, pathlib
from model import EconomyModel
from climada.util.constants import HAZ_DEMO_FL

def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--hazard-file", default=HAZ_DEMO_FL)
    p.add_argument("--hazard-year", type=int)
    p.add_argument("--viz", action="store_true", help="Launch interactive Solara dashboard instead of headless run")
    return p.parse_args()

def main() -> None:  # noqa: D401
    args = _parse()

    # If visualization requested, delegate to Solara which hosts the dashboard
    if args.viz:
        import subprocess, sys, os

        env = os.environ.copy()
        env["ABM_HAZARD_FILE"] = str(args.hazard_file)
        if args.hazard_year is not None:
            env["ABM_HAZARD_YEAR"] = str(args.hazard_year)

        cmd = [sys.executable, "-m", "solara", "run", "visualization.py"]
        subprocess.run(cmd, env=env, check=False)
        return

    model = EconomyModel(
        width=10,
        height=10,
        num_households=100,
        num_firms=20,
        shock_step=5,
        scenario="demo",                # anything but "synthetic"
        hazard_file=str(pathlib.Path(args.hazard_file).expanduser()),
        hazard_year=args.hazard_year,
    )

    for _ in range(10):  # simulate 10 years
        model.step()

    model.save_results("simulation_results")
    print("Simulation complete. Results stored in simulation_results.csv")


if __name__ == "__main__":
    main() 