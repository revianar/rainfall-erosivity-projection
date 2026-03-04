import xarray as xr
import numpy as np
from pathlib import Path
import click
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Jakarta bounding box ──────────────────────────────────────────────────────
JAKARTA_BBOX = {
    "lat_min": -7.0,
    "lat_max": -5.5,
    "lon_min": 106.0,
    "lon_max": 107.5,
}

SCENARIOS = ["historical", "rcp26", "rcp45", "rcp85"]
MODEL = "HadGEM2-AO"


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
    logger.info("Processing: %s", input_path.name)   
    ds = xr.open_dataset(input_path, use_cftime=True)

    # Rename coordinates if needed (some models use 'latitude'/'longitude')
    rename_map = {}
    if "latitude" in ds.coords and "lat" not in ds.coords:
        rename_map["latitude"] = "lat"
    if "longitude" in ds.coords and "lon" not in ds.coords:
        rename_map["longitude"] = "lon"
    if rename_map:
        ds = ds.rename(rename_map)

    ds = crop_to_jakarta(ds, bbox)
    ds = convert_pr_units(ds, pr_var)

    # Update global attributes
    ds.attrs["domain"] = "Jakarta Greater Capital Region (Jabodetabek)"
    ds.attrs["processing"] = "Cropped and unit-converted from CMIP5 raw data"

    output_path = output_dir / f"{input_path.stem}_jakarta.nc"
    ds.to_netcdf(output_path)
    logger.info(f"Saved to: {output_path}")
    ds.close()
    return output_path


@click.command()
@click.option(
    "--input-dir",
    default="esgf",
    show_default=True,
    help="Directory containing raw CMIP5 NetCDF files.",
)
@click.option(
    "--output-dir",
    default="esgf/processed",
    show_default=True,
    help="Directory to save cropped files.",
)
@click.option(
    "--scenario",
    default="all",
    type=click.Choice(SCENARIOS + ["all"]),
    show_default=True,
    help="Which RCP scenario to process.",
)
@click.option(
    "--pr-var",
    default="pr",
    show_default=True,
    help="Name of the precipitation variable in the NetCDF file.",
)
def main(input_dir: str, output_dir: str, scenario: str, pr_var: str):
    """Crop CMIP5 HadGEM2-AO precipitation files to the Jakarta domain."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    scenarios_to_process = SCENARIOS if scenario == "all" else [scenario]

    for scen in scenarios_to_process:
        pattern = f"*{MODEL}*{scen}*pr*.nc"
        files = sorted(input_path.glob(pattern))
        if not files:
            logger.warning(f"No files found matching pattern: {pattern}")
            continue
        for f in files:
            process_file(f, output_path, pr_var=pr_var)

    logger.info("Domain cropping complete.")


if __name__ == "__main__":
    main()
