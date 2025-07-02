# Spatial economic ABM for climate risk

This repository provides a **spatial agent-based model** (ABM) that couples basic economic behaviour with climate-hazard impacts. It is built with [Mesa](https://mesa.readthedocs.io) and is designed to plug into [CLIMADA](https://github.com/CLIMADA-project/climada_python) as the climate-impact engine.

## Features included

* Agents distributed on spatial grid corresponding to the input climate data file (`mesa.space.MultiGrid`).
* `HouseholdAgent` and `FirmAgent` instances randomly distributed (number provided by user).
* CLIMADA hazard data used for input climate shocks.
* Migration by households when local risk exceeds a threshold.
* Firm production reduced in proportion to local hazard intensity.
* Collection of yearly GDP and migrant counts.
* Results saved to `simulation_results.csv`.
* **live dashboard** built with Mesa's Solara API – launch with `--viz` to watch the grid and charts update in real time.

## Quick start

### 1. Create and activate a virtual environment (optional but recommended)
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the model

Head-less run (CSV output only):

```bash
python run_simulation.py --hazard-file demo_hist.hdf5
```

Interactive dashboard (live grid + plots):

```bash
python run_simulation.py --viz --hazard-file demo_hist.hdf5
```

The command opens a local Solara web-app (default `http://localhost:8765`).  Use the sliders to tweak agents, shock year etc. while the simulation is running.

```
$ cat simulation_results.csv
GDP,Migrants,Average_Risk
77.2,0,0.0
77.5,0,0.0
…
```

## Next steps
* Extend agent behaviour (production, labour, trade, ...).
* Better climate data and addition of vulnerabilities.
* Export regional metrics (GDP per cell, Gini index, recovery time, etc.).

## Using real CLIMADA hazards

1. Generate or obtain a `Hazard` object for your desired SSP/RCP scenario and year.  With CLIMADA this is typically done in a Python session:

```python
##############################################
# 1) Quick demo hazards shipped with CLIMADA #
##############################################

# Newer versions (≥5.x) expose the demo HDF5 paths via constants
from climada.hazard import Hazard
from climada.util.constants import HAZ_DEMO_H5, HAZ_DEMO_FL

# Load one of the built-in demo hazards and save a local copy you can tweak
# – HAZ_DEMO_H5 is a generic historical multi-hazard sample (already plain HDF5)
# – HAZ_DEMO_FL is a **gzipped** flood hazard (…hdf5.gz). The model will
#   *decompress it on the fly*, but if you prefer to have the plain file you can
#   gunzip it yourself first:
#     gunzip "$HAZ_DEMO_FL"   # creates the .hdf5 next to it
#   or in Python:
#     import gzip, shutil, pathlib, os
#     gz = pathlib.Path(HAZ_DEMO_FL)
#     with gzip.open(gz, 'rb') as f_in, open(gz.with_suffix(''), 'wb') as f_out:
#         shutil.copyfileobj(f_in, f_out)

hax_source = HAZ_DEMO_FL  # or HAZ_DEMO_H5, HAZ_DEMO_MAT, …
haz = Hazard.from_hdf5(hax_source if hax_source.endswith('.hdf5') else hax_source.rstrip('.gz'))
haz.write_hdf5('demo_hist.hdf5')

# Tip: to see all available demo files just list the constants:
#   >>> import climada.util.constants as c
#   >>> [k for k in dir(c) if k.startswith('HAZ_DEMO')]

############################################################
# 2) Download a scenario-specific set from the Data-API     #
############################################################

from climada.util.api_client import download_hazard  # requires climada>=5.0

haz = download_hazard(
    hazard_type='TC',      # tropical cyclone wind
    scenario='ssp245',     # SSP2-RCP4.5
    year=2050,
    file_name='tc_ssp245_2050.hdf5'
)
```

2. Run the ABM and point it to the file:

```bash
python run_simulation.py --hazard-file tc_ssp245_2050.hdf5 --hazard-year 2050
```
(Or hard-code the arguments in `run_simulation.py`.)

During runtime the console will print a clear `[INFO] Loaded CLIMADA hazard from ...` message.  If anything goes wrong you'll see a `[WARNING] Falling back to synthetic hazard ...` message, so you always know which data source is active. 

If you pass neither a `--hazard-file` argument nor the `hazard_file=` parameter the loader now:

* automatically locates the built-in CLIMADA demo file (`HAZ_DEMO_TC` or `HAZ_DEMO_FL`).
* if that file is shipped as a **gzipped** HDF5 (`*.hdf5.gz`), the model now *automatically
  decompresses* it the first time you run the simulation. This fixes the
  "Unable to synchronously open file (file signature not found)" error you might
  have seen when CLIMADA >=5 only provided the compressed demo data.

You can still point to any custom hazard with

```bash
python run_simulation.py --hazard-file "/path/to/my.hdf5" --hazard-year 2050
``` 