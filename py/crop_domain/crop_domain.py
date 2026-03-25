import sys
import xarray as xr
import numpy as np
from pathlib import Path
import click
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT    = PROJECT_ROOT.parent

JAKARTA_BBOX = {
    "lat_min": -8.75,
    "lat_max": -3.75,
    "lon_min": 103.125,
    "lon_max": 111.875,
}

MODELS = {
    "MRI-ESM2-0": {
        "ensemble":  "r1i1p1f1",
        "grid":      "gn",
        "scenarios": ["historical", "ssp126", "ssp245", "ssp585"],
    },
    "EC-Earth3": {
        "ensemble":  "r1i1p1f1",
        "grid":      "gr",
        "scenarios": ["historical", "ssp126", "ssp245", "ssp585"],
    },
    "CNRM-CM6-1": {
        "ensemble":  "r1i1p1f2",
        "grid":      "gr",
        "scenarios": ["historical", "ssp126", "ssp245", "ssp585"],
    },
}

ALL_SCENARIOS = ["historical", "ssp126", "ssp245", "ssp585"]

HIST_START = 1950
HIST_END   = 2014
PROJ_START = 2015
PROJ_END   = 2100


# ===== Core helpers ====================

def crop_to_jakarta(ds: xr.Dataset, bbox: dict = JAKARTA_BBOX) -> xr.Dataset:
    if float(ds.lon.values.max()) > 180:
        logger.info("  Detected 0-360 longitude. Converting to -180-180.")
        ds = ds.assign_coords(lon=((ds.lon + 180) % 360) - 180)
        ds = ds.sortby("lon")

    ds_cropped = ds.sel(
        lat=slice(bbox["lat_min"], bbox["lat_max"]),
        lon=slice(bbox["lon_min"], bbox["lon_max"]),
    )
    logger.info(
        f"  Domain: lat [{float(ds_cropped.lat.min()):.2f}, {float(ds_cropped.lat.max()):.2f}]  "
        f"lon [{float(ds_cropped.lon.min()):.2f}, {float(ds_cropped.lon.max()):.2f}]  "
        f"({len(ds_cropped.lat)} x {len(ds_cropped.lon)} cells)"
    )
    return ds_cropped


def convert_pr_units(ds: xr.Dataset, pr_var: str = "pr") -> xr.Dataset:
    units = ds[pr_var].attrs.get("units", "")
    if units in ["kg m-2 s-1", "kg/m2/s", "kg m-2 s-1 "]:
        logger.info("  Converting units: kg m-2 s-1 -> mm/day")
        ds[pr_var] = ds[pr_var] * 86400.0
        ds[pr_var].attrs["units"] = "mm/day"
    elif units == "mm/day":
        logger.info("  Units already mm/day. nNo conversion needed.")
    else:
        logger.warning(f"  Unrecognised units '{units}'. Skipping conversion.")
    return ds


def normalise_coords(ds: xr.Dataset) -> xr.Dataset:
    rename_map = {}
    if "latitude"  in ds.coords and "lat" not in ds.coords:
        rename_map["latitude"]  = "lat"
    if "longitude" in ds.coords and "lon" not in ds.coords:
        rename_map["longitude"] = "lon"
    if rename_map:
        logger.info(f"  Renaming coords: {rename_map}")
        ds = ds.rename(rename_map)
    return ds


# ===== File discovery ====================

def find_chunks(input_dir: Path, model: str, scenario: str) -> list:
    """
    CMIP6 filenames follow the pattern:
        pr_day_<model>_<scenario>_<ensemble>_<grid>_<start>-<end>.nc
    """
    cfg     = MODELS[model]
    pattern = f"pr_day_{model}_{scenario}_{cfg['ensemble']}_{cfg['grid']}_*.nc"
    chunks  = sorted(input_dir.glob(pattern))
    return chunks


# ===== Single-scenario processor ====================

def process_scenario(
    input_dir:  Path,
    output_dir: Path,
    model:      str,
    scenario:   str,
    bbox:       dict = JAKARTA_BBOX,
    pr_var:     str  = "pr",
):
    """
    Discover all chunk files for model+scenario, open as a single dataset via open_mfdataset,
    crop, convert units, and write one merged output file.
    """
    cfg    = MODELS[model]
    chunks = find_chunks(input_dir, model, scenario)

    if not chunks:
        logger.warning(f"  No files found for {model} / {scenario} -- skipping.")
        return None

    logger.info(f"  Found {len(chunks)} chunk(s):")
    for c in chunks:
        logger.info(f"    {c.name}")

    # Open all chunks as one dataset (cftime-safe)
    logger.info("  Merging chunks with open_mfdataset ...")
    ds = xr.open_mfdataset(
        chunks,
        combine="by_coords",
        decode_times=xr.coders.CFDatetimeCoder(use_cftime=True),
        chunks={"time": 365},   # lazy loading
    )

    ds = normalise_coords(ds)
    ds = crop_to_jakarta(ds, bbox)
    ds = convert_pr_units(ds, pr_var)

    # Attach provenance metadata
    ds.attrs.update({
        "model":        model,
        "scenario":     scenario,
        "ensemble":     cfg["ensemble"],
        "grid":         cfg["grid"],
        "domain":       "Jakarta Greater Capital Region",
        "processing":   "Merged CMIP6 chunks, cropped to Jakarta domain, units converted to mm/day",
        "bbox":         (f"lat [{bbox['lat_min']}, {bbox['lat_max']}], "
                         f"lon [{bbox['lon_min']}, {bbox['lon_max']}]"),
        "source_files": ", ".join(c.name for c in chunks),
    })

    out_name = f"pr_day_{model}_{scenario}_{cfg['ensemble']}_jakarta.nc"
    out_path = output_dir / out_name

    logger.info(f"  Writing -> {out_name}")
    ds.to_netcdf(
        out_path,
        encoding={pr_var: {"dtype": "float32", "zlib": True, "complevel": 4}},
    )
    ds.close()

    # Debugging on the written file
    ds_check = xr.open_dataset(
        out_path,
        decode_times=xr.coders.CFDatetimeCoder(use_cftime=True),
    )
    t0 = str(ds_check.time.values[0])[:10]
    t1 = str(ds_check.time.values[-1])[:10]
    n  = len(ds_check.time)
    pr = ds_check[pr_var]
    logger.info(
        f"  Saved  period={t0} -> {t1}  days={n:,}  "
        f"mean={float(pr.mean()):.2f} mm/day  max={float(pr.max()):.2f} mm/day"
    )
    ds_check.close()
    return out_path


# ===== CLI ======================================================================

@click.command()
@click.option(
    "--input-dir",
    default=str(DATA_ROOT / "CMIP6"),
    show_default=True,
    help="Directory containing raw CMIP6 NetCDF files.",
)
@click.option(
    "--output-dir",
    default=str(DATA_ROOT / "Rainfall-Erosivity" / "py" / "data" / "processed"),
    show_default=True,
    help="Directory to save merged + cropped output files.",
)
@click.option(
    "--model",
    default="all",
    type=click.Choice(list(MODELS.keys()) + ["all"]),
    show_default=True,
    help="Which CMIP6 model to process, or 'all'.",
)
@click.option(
    "--scenario",
    default="all",
    type=click.Choice(ALL_SCENARIOS + ["all"]),
    show_default=True,
    help="Which scenario to process, or 'all'.",
)
@click.option(
    "--pr-var",
    default="pr",
    show_default=True,
    help="Name of the precipitation variable inside the NetCDF files.",
)
def main(input_dir: str, output_dir: str, model: str, scenario: str, pr_var: str):
    input_path  = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        logger.error(f"Input directory not found: {input_path}")
        logger.error("Place your CMIP6 .nc files in the CMIP/ folder.")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("CMIP6 crop_domain.py")
    logger.info("=" * 60)
    logger.info(f"Project root : {PROJECT_ROOT}")
    logger.info(f"Input dir    : {input_path}")
    logger.info(f"Output dir   : {output_path}")

    models_to_run    = list(MODELS.keys()) if model    == "all" else [model]
    scenarios_to_run = ALL_SCENARIOS       if scenario == "all" else [scenario]

    completed = []
    skipped   = []

    for mdl in models_to_run:
        for scen in scenarios_to_run:
            # Only process scenarios defined for this model
            if scen not in MODELS[mdl]["scenarios"]:
                logger.info(f"\nSkipping {mdl} / {scen} -- not in model scenario list.")
                skipped.append((mdl, scen))
                continue

            logger.info(f"{'=' * 60}")
            logger.info(f"Model: {mdl}  |  Scenario: {scen}")
            logger.info(f"{'=' * 60}")

            out = process_scenario(
                input_dir  = input_path,
                output_dir = output_path,
                model      = mdl,
                scenario   = scen,
                pr_var     = pr_var,
            )
            if out:
                completed.append((mdl, scen, out.name))
            else:
                skipped.append((mdl, scen))

    logger.info("\n" + "=" * 60)
    logger.info(f"DONE -- {len(completed)} file(s) written, {len(skipped)} skipped")
    logger.info("=" * 60)
    for mdl, scen, fname in completed:
        logger.info(f"  OK  {mdl:<15} {scen:<12} -> {fname}")
    if skipped:
        logger.info("Skipped (no files found or not in model list):")
        for item in skipped:
            logger.info(f"  --  {item[0]:<15} {item[1]}")


if __name__ == "__main__":
    main()