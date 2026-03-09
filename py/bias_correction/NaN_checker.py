'''
There has to be 0 (zero) NaNs because I applied the bias correction transfer function from the nearest land cell.
So the statistical relationship between CHIRPS observations and the model's historical simulation,
where it learned at a land cell, was spatially extrapolated to correct the model's precipitation at adjacent ocean cells.

Extra notes for transparency on the methodology:
1. HadGEM2-AO at 1.25°×1.875° resolves Java with only ~8 land cells — this is a known limitation of coarse GCMs for island/coastal domains,
2. Ocean cells were bias-corrected using transfer functions from the nearest land cell — this is a spatial extrapolation assumption,
3. Results over ocean cells should be interpreted as indicative of the regional precipitation regime, not as point estimates for those specific locations.
'''

import xarray as xr
import numpy as np

ds = xr.open_dataset(
    r'D:\Tugas\Personal\Jupyter Notebook\Rainfall-Erosivity\py\esgf\bias_corrected\pr_day_HadGEM2-AO_rcp85_r1i1p1_jakarta_bc_empirical.nc',
    decode_times=xr.coders.CFDatetimeCoder(use_cftime=True)
)
print('pr shape:', ds['pr'].values.shape)
print('pr NaNs: ', int(np.isnan(ds['pr'].values).sum()))
print('pr mean: ', float(np.nanmean(ds['pr'].values)))
ds.close()
