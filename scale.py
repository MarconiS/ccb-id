import sys

import numpy as np
import pandas as pd

import ccbid
from ccbid import args
from ccbid import prnt

pt = "/Users/sergiomarconi/"
ainput = "NIS1_20140507_144927_atmcor_CNJRF.tif"
aitcput = "/Users/sergiomarconi/"+ainput[:-4]+".shp"
#ainput = args[1]
aremove_outliers = 'PCA'
athreshold = 3
aaggregate = 'average'
# args.feature_selection = False


np.random.seed(1984)
model = ccbid.read.pck("./model/ensemble")

# get base data from the model
sp_labels = model.labels_

from skimage import io
features = np.array(io.imread(pt+ainput))
npix = features.shape
features = features.flatten().reshape(npix[0]*npix[1], 369)

from skimage import io
features = np.array(io.imread(pt+ainput))
npix = features.shape
features = features.flatten().reshape(npix[0]*npix[1], 369)

mask1 = np.all(features < 0, axis=1) | np.all(features > 10000, axis=1)
data = features[~mask1]
n_ok = data.shape


mask_pca = ccbid.outliers.with_pca(data, thresh=athreshold)
data = data[mask_pca, 24:369]
#id_labels = id_labels[mask]

data = model.reducer.transform(data)
data = data[:, 0:model.n_features_]
prnt.status("Applying CCBID model to input features")
prob = model.predict_proba(data, average_proba=True)

output = pd.DataFrame(index=range(mask_pca.shape[0]), columns=range(prob.shape[1]))

output.loc[mask_pca,:] =  prob
final = pd.DataFrame(index=range(features.shape[0]), columns=range(prob.shape[1]))
final[~mask1] = output

#get the crown ids from the crown delinetation
import geopandas as gpd
import rasterio
from rasterio import features

itc = gpd.read_file(aitcput)

final = final.values.reshape(npix[0], npix[1], final.shape[1])
with rasterio.open(ainput) as dataset:
    extent = dataset.transform
    data_crs = dataset.crs



from rasterio.transform import from_origin
final = final.astype(float)

new_dataset = rasterio.open('test1.tif', 'w', driver='GTiff',
                            height = final.shape[0], width = final.shape[1],
                            count=prob.shape[1], dtype=str(final.dtype),
                            crs=data_crs,
                            transform=extent)

for ii in range(prob.shape[1]):
    tmp = final[:,:,ii].astype(float)
    new_dataset.write(tmp, ii+1)
new_dataset.close()

from rasterstats import zonal_stats
#get the crown ids from the crown delinetation
import geopandas as gpd
import rasterio
from rasterio import features

itc = gpd.read_file(aitcput)
itc.crs = data_crs

for ii in range(prob.shape[1]):
    with rasterio.open("./polygons/rasters/"+ainput) as src:
        affine = src.transform
        array = src.read(ii+1)
        df_zonal_stats = pd.DataFrame(zonal_stats(itc, array, affine=affine, stats=['mean']))

    # adding statistics back to original GeoDataFrame
    itc = pd.concat([itc, df_zonal_stats], axis=1) 



csv_name = "./predictions/polygons/"+ainput[:-4]
itc.dropna().to_csv(csv_name+".csv", index=False) 







