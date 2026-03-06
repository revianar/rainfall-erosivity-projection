import sys
import xarray as xr
import numpy as np
from xarray.coding.times import CFDatetimeCoder
from pathlib import Path
import click
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ===== Jakarta bounding box ====================
JAKARTA_BBOX = {
    "lat_min": -7.0,
    "lat_max": -5.5,
    "lon_min": 106.0,
    "lon_max": 107.5,
}

MODEL = "HadGEM2-AO"                                      # Interchangable (e.g., MPI-ESM-MR, IPSL-CMSA-LR)
SCENARIOS = ["historical", "rcp26", "rcp45", "rcp85"]     # Interchangable (e.g., rcp60)


def crop_to_jakarta(ds: xr.Dataset, bbox: dict = JAKARTA_BBOX) -> xr.Dataset:
    # Normalize longitudes if in 0–360° convention
    if ds.lon.values.max() > 180:
        logger.info("Detected 0–360° longitude convention. Converting to -180–180°.")
        ds = ds.assign_coords(lon=((ds.lon + 180) % 360) - 180)
        ds = ds.sortby("lon")

    ds_cropped = ds.sel(
        lat=slice(bbox["lat_min"], bbox["lat_max"]),
        lon=slice(bbox["lon_min"], bbox["lon_max"]),
    )
    logger.info(
        f"Cropped domain: lat [{ds_cropped.lat.values.min():.2f}, "
        f"{ds_cropped.lat.values.max():.2f}], "
        f"lon [{ds_cropped.lon.values.min():.2f}, "
        f"{ds_cropped.lon.values.max():.2f}]"
    )
    return ds_cropped


def convert_pr_units(ds: xr.Dataset, pr_var: str = "pr") -> xr.Dataset:
    if ds[pr_var].attrs.get("units", "") in ["kg m-2 s-1", "kg/m2/s"]:
        logger.info("Converting precipitation units: kg m⁻² s⁻¹ → mm/day")
        ds[pr_var] = ds[pr_var] * 86400.0
        ds[pr_var].attrs["units"] = "mm/day"
    return ds


def process_file(
    input_path: Path,
    output_dir: Path,
    bbox: dict = JAKARTA_BBOX,
    pr_var: str = "pr",
) -> Path:
    logger.info(f"Processing: {input_path.name}")
    ds = xr.open_dataset(input_path, decode_times=CFDatetimeCoder(use_cftime=True))

    # Rename coordinate aliases (some models use 'latitude'/'longitude')
    rename_map = {}
    if "latitude" in ds.coords and "lat" not in ds.coords:
        rename_map["latitude"] = "lat"
    if "longitude" in ds.coords and "lon" not in ds.coords:
        rename_map["longitude"] = "lon"
    if rename_map:
        logger.info(f"Renaming coordinates: {rename_map}")
        ds = ds.rename(rename_map)

    ds = crop_to_jakarta(ds, bbox)
    ds = convert_pr_units(ds, pr_var)

    # Update global attributes
    ds.attrs["domain"]     = "Jakarta Greater Capital Region (Jabodetabek)"
    ds.attrs["processing"] = "Cropped and unit-converted from CMIP5 raw data"
    ds.attrs["bbox"]       = (f"lat [{bbox['lat_min']}, {bbox['lat_max']}], "
                               f"lon [{bbox['lon_min']}, {bbox['lon_max']}]")

    output_path = output_dir / f"{input_path.stem}_jakarta.nc"
    ds.to_netcdf(output_path)
    logger.info(f"Saved to: {output_path}")
    ds.close()
    return output_path


@click.command()
@click.option(
    "--input-dir",
    # Default resolves to Rainfall-Erosivity/py/esgf/ regardless of where the script is called from
    default=str(PROJECT_ROOT / "py" / "esgf"),
    show_default=True,
    help="Directory containing raw CMIP5 NetCDF files.",
)
@click.option(
    "--output-dir",
    default=str(PROJECT_ROOT / "py" / "esgf" / "processed"),
    show_default=True,
    help="Directory to save cropped files.",
)
@click.option(
    "--scenario",
    default="all",
    type=click.Choice(SCENARIOS + ["all"]),
    show_default=True,
    help="Which RCP scenario to process (or 'all' for every scenario).",
)
@click.option(
    "--pr-var",
    default="pr",
    show_default=True,
    help="Name of the precipitation variable inside the NetCDF file.",
)
def main(input_dir: str, output_dir: str, scenario: str, pr_var: str):
    input_path  = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        logger.error(f"Input directory not found: {input_path}")
        logger.error("Make sure your CMIP5 .nc files are placed in esgf")
        sys.exit(1)

    logger.info(f"Project root : {PROJECT_ROOT}")
    logger.info(f"Input dir    : {input_path}")
    logger.info(f"Output dir   : {output_path}")

    scenarios_to_process = SCENARIOS if scenario == "all" else [scenario]

    total_processed = 0
    for scen in scenarios_to_process:
        pattern = f"pr_day_{MODEL}_{scen}_*.nc"
        files = sorted(input_path.glob(pattern))
        if not files:
            logger.warning(f"No files found matching pattern: {pattern}")
            continue
        logger.info(f"\nScenario '{scen}': {len(files)} file(s) found")
        for f in files:
            process_file(f, output_path, pr_var=pr_var)
            total_processed += 1

    if total_processed == 0:
        logger.warning(
            f"'{MODEL}' and the scenario name (e.g., historical, rcp85)."
        )
    else:
        logger.info(f"\nDomain cropping complete. {total_processed} file(s) saved to {output_path}")


if __name__ == "__main__":
    main()