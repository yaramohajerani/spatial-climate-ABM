"""Run prototype climate-economy ABM for 10 timesteps and persist results."""

import argparse
from model import EconomyModel
# Runner now expects one or more --rp-file arguments in the form
# "<RP>:<START_STEP>:<END_STEP>:<TYPE>:<path>"

def _parse():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--rp-file",
        action="append",
        metavar="RP:START:END:TYPE:PATH",
        help=(
            "Add a GeoTIFF file. Format: <RP>:<START_STEP>:<END_STEP>:<HAZARD_TYPE>:<path>. "
            "Required unless provided via --param-file. "
            "Example: --rp-file 100:1:20:FL:rp100_2030.tif"
        ),
    )
    p.add_argument("--viz", action="store_true", help="Launch interactive Solara dashboard instead of headless run")
    p.add_argument("--steps", type=int, default=10, help="Number of timesteps to simulate")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    p.add_argument("--start-year", type=int, default=0, help="Base calendar year for step 0 (optional; used for plotting)")
    p.add_argument("--topology", type=str, help="Optional JSON file describing firm supply-chain topology")
    p.add_argument(
        "--param-file",
        type=str,
        help="Path to a JSON file containing parameter overrides. Keys can include rp_files (list), viz (bool), seed (int), topology (str).",
    )
    return p.parse_args()

def main() -> None:  # noqa: D401
    args = _parse()

    # ---------------- Optional parameter file ---------------------------- #
    if args.param_file:
        import json, pathlib

        param_path = pathlib.Path(args.param_file)
        if not param_path.exists():
            raise SystemExit(f"Parameter file not found: {param_path}")

        try:
            with param_path.open() as f:
                param_data = json.load(f)
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(f"Failed to parse JSON parameter file: {param_path}") from exc

        # Merge parameters – CLI flags take precedence over file settings
        # 1. rp_files (list[str]) ----------------------------------------
        file_rp = param_data.get("rp_files") or param_data.get("rp_file")
        if file_rp and not args.rp_file:
            args.rp_file = file_rp
        elif file_rp and args.rp_file:
            # Combine – keep CLI order last so they override duplicates
            args.rp_file = file_rp + args.rp_file

        # 2. Viz flag ------------------------------------------------------
        if getattr(args, "viz", False) is False and param_data.get("viz"):
            args.viz = bool(param_data.get("viz"))

        # 3. Seed -----------------------------------------------------------
        if args.seed == 42 and "seed" in param_data:
            args.seed = int(param_data["seed"])

        # 3a. Steps ---------------------------------------------------------
        if args.steps == 10 and "steps" in param_data:
            args.steps = int(param_data["steps"])

        # 3b. Start year ----------------------------------------------------
        if args.start_year == 0 and "start_year" in param_data:
            args.start_year = int(param_data["start_year"])

        # 4. Topology path --------------------------------------------------
        if not args.topology and param_data.get("topology"):
            args.topology = str(param_data["topology"])

    # Ensure we have at least one RP spec after merging param file -----------
    if not args.rp_file:
        raise SystemExit("No --rp-file entries provided and none found in parameter file.")

    # First, parse the RP files into a list irrespective of --viz so we can
    # pass them on to a potential Solara dashboard.
    # Parsed as (return_period, start_step, end_step, hazard_type, path)
    events: list[tuple[int, int, int, str, str]] = []
    if args.rp_file:
        for item in args.rp_file:
            try:
                rp_str, start_str, end_str, type_str, path_str = item.split(":", 4)
                events.append((int(rp_str), int(start_str), int(end_str), type_str, path_str))
            except ValueError as exc:  # noqa: BLE001
                raise SystemExit(
                    (
                        f"Invalid --rp-file format: {item}. Expected "
                        "<RP>:<START>:<END>:<TYPE>:<path>."
                    )
                ) from exc

    # If visualization requested, delegate to Solara which hosts the dashboard
    if args.viz:
        import subprocess, sys, os

        env = os.environ.copy()
        # Pass hazard events to the dashboard so it can build the same model
        env["ABM_HAZARD_EVENTS"] = ";".join(
            f"{rp}:{s}:{e}:{t}:{p}" for rp, s, e, t, p in events
        )
        env["ABM_SEED"] = str(args.seed)
        if args.topology:
            env["ABM_TOPOLOGY_PATH"] = args.topology

        if args.start_year:
            env["ABM_START_YEAR"] = str(args.start_year)

        cmd = [sys.executable, "-m", "solara", "run", "visualization.py"]
        subprocess.run(cmd, env=env, check=False)
        return

    # Headless mode: run the simulation directly
    model = EconomyModel(
        num_households=100,
        num_firms=20,
        shock_step=5,
        hazard_events=events,
        seed=args.seed,
        firm_topology_path=args.topology,
        start_year=args.start_year,
    )

    for _ in range(args.steps):
        model.step()

    model.save_results("simulation_results")

    # ------------------------- Plotting ---------------------------- #
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    df = model.results_to_dataframe()

    rename_map = {"Base_Wage": "Mean_Wage", "Household_LaborSold": "Household_Labor_Sold"}
    df = df.rename(columns=rename_map)

    units = {
        "Firm_Production": "Units of Goods",
        "Firm_Consumption": "Units of Goods",
        "Firm_Wealth": "$",
        "Firm_Capital": "Units of Capital",
        "Household_Wealth": "$",
        "Household_Capital": "Units of Capital",
        "Household_Labor_Sold": "Units of Labor",
        "Household_Consumption": "Units of Goods",
        "Average_Risk": "Score (0–1)",
        "Mean_Wage": "$ / Unit of Labor",
        "Mean_Price": "$ / Unit of Goods",
        "Labor_Limited_Firms": "count",
        "Capital_Limited_Firms": "count",
        "Input_Limited_Firms": "count",
    }

    firm_metrics = [
        "Firm_Production",
        "Firm_Consumption",
        "Firm_Wealth",
        "Firm_Capital",
        "Mean_Price",
    ]

    household_metrics = [
        "Household_Labor_Sold",
        "Household_Consumption",
        "Household_Wealth",
        "Mean_Wage",
        "Average_Risk",
    ]

    rows = max(len(firm_metrics), len(household_metrics)) + 1  # extra row for bottlenecks
    fig = plt.figure(figsize=(10, rows * 3))
    gs = gridspec.GridSpec(rows, 2, height_ratios=[1]*(rows-1)+[1.2])
    axes_matrix = [[fig.add_subplot(gs[r, c]) for c in range(2)] for r in range(rows)]

    x_col = "Year" if args.start_year else "Step"
    if args.start_year:
        df["Year"] = args.start_year + df.index.astype(int) / 4

    def _plot(col, ax):
        ax.plot(df[x_col], df[col])
        ax.set_title(col.replace("_", " "), fontsize=10)
        ylabel = units.get(col, "")
        if ylabel:
            ax.set_ylabel(ylabel)
        ax.set_xlabel(x_col)

    for r in range(rows-1):
        if r < len(firm_metrics):
            _plot(firm_metrics[r], axes_matrix[r][0])
        else:
            axes_matrix[r][0].set_visible(False)

        if r < len(household_metrics):
            _plot(household_metrics[r], axes_matrix[r][1])
        else:
            axes_matrix[r][1].set_visible(False)

    # bottlenecks row
    ax_bott = fig.add_subplot(gs[-1, :])
    for col, color in [
        ("Labor_Limited_Firms", "tab:blue"),
        ("Capital_Limited_Firms", "tab:red"),
        ("Input_Limited_Firms", "tab:green"),
    ]:
        if col in df.columns:
            ax_bott.plot(df[x_col], df[col], label=col, color=color)
    ax_bott.set_title("Production Bottlenecks", fontsize=10)
    ax_bott.set_ylabel("count")
    ax_bott.set_xlabel(x_col)
    ax_bott.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig("simulation_timeseries.png", dpi=150)
    plt.close(fig)

    print("Simulation complete. Results stored in simulation_results.csv and simulation_timeseries.png")


if __name__ == "__main__":
    main() 