import xarray as xr
import dask.array as da
import numpy as np
import matplotlib.pyplot as plt
import glob as glob
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd

# Read in forecast (All initialization times, 12 hour lead time)
fc_files = sorted(glob.glob('/Users/dcalhoun/Desktop/Research/Data/ecmwf/forecast_2t_12hr_2016_2024/*/*/*.nc'))
ds_fc = xr.open_mfdataset(fc_files[:62], combine='by_coords', parallel=True)

# Ensure time coordinate is sorted chronologically
ds_fc = ds_fc.sortby('time')
t2m = ds_fc.t2m - 273.15  # Convert from Kelvin to Celsius

print(type(t2m))
print(t2m)
print(t2m.shape)

t2m_raw = t2m.values # extract the temperature data as a raw numpy/dask array, losing lat/lon coordinate and time metadata
print(type(t2m_raw))
print(t2m_raw)
print(t2m_raw.shape)

# Basic plotting of the mean temperature forecast
fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()})
mesh = t2m.mean(axis=0).plot(ax=ax, cmap='RdBu', add_colorbar=False)
ax.add_feature(cfeature.BORDERS)
ax.add_feature(cfeature.COASTLINE)
ax.set_aspect(1.5)
cbar = fig.colorbar(mesh, ax=ax, orientation='horizontal', fraction=0.046, pad=0.04)
cbar.set_label('C')
plt.title('Mean 2m Temperature Forecast (C), January 2016, 12hr Lead Time')
plt.show()

# Cleaned-up plotting of two forecasts on the same color scale with formatted titles
day1 = t2m.sel(time="2016-01-01T12:00:00")
day2 = t2m.isel(time=21)

# compute a common color scale
vals = np.concatenate([day1.values.ravel(), day2.values.ravel()])
vmin, vmax = np.nanmin(vals), np.nanmax(vals)

fig, axs = plt.subplots(2, 1, figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})

mesh1 = day1.plot(ax=axs[0], cmap='RdBu_r', add_colorbar=False, vmin=vmin, vmax=vmax)
axs[0].add_feature(cfeature.BORDERS)
axs[0].add_feature(cfeature.COASTLINE)
axs[0].set_aspect(1.5)
t1 = pd.to_datetime(day1.time.values).strftime('%Y-%m-%d %H:%M UTC')
axs[0].set_title(f'12-hr 2m Temperature Forecast (°C), valid {t1}')

mesh2 = day2.plot(ax=axs[1], cmap='RdBu_r', add_colorbar=False, vmin=vmin, vmax=vmax)
axs[1].add_feature(cfeature.BORDERS)
axs[1].add_feature(cfeature.COASTLINE)
axs[1].set_aspect(1.5)
t2 = pd.to_datetime(day2.time.values).strftime('%Y-%m-%d %H:%M UTC')
axs[1].set_title(f'12-hr 2m Temperature Forecast (°C), valid {t2}')

# per-panel horizontal colorbars
cbar1 = fig.colorbar(mesh1, ax=axs[0], orientation='horizontal', fraction=0.046, pad=0.04)
cbar1.set_label('°C')
cbar2 = fig.colorbar(mesh2, ax=axs[1], orientation='horizontal', fraction=0.046, pad=0.04)
cbar2.set_label('°C')

plt.tight_layout()
plt.show()