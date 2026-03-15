import rasterio
import pandas as pd
import numpy as np

# Path to the TIF file
tif_path = "/home/prajwalmac/7th Semester/Project/datas/organic/organic.tif"

# Open raster and read data
with rasterio.open(tif_path) as src:
    data = src.read(1)          # read first band
    nodata_val = src.nodata     # save nodata value before closing
    data = data.flatten()       # flatten 2D → 1D

# Create DataFrame
df = pd.DataFrame({"organic": data})

# Remove no-data values
if nodata_val is not None:      # only replace if nodata is defined
    df = df.replace(nodata_val, np.nan)

df.dropna(inplace=True)

# Save to CSV
df.to_csv("/home/prajwalmac/7th Semester/Project/datas/organic_data.csv", index=False)

# Preview
print(df.head())
print("Total samples:", len(df))
