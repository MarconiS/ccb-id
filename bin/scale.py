import sys

import numpy as np
import pandas as pd

import ccbid
from ccbid import args
from ccbid import prnt

pt = "/orange/ewhite/NeonData/2015_Campaign/D03/OSBS/L5/Rasters/"
ainput = "NIS1_20140507_144927_atmcor_CNJRF.tif"
#aitcput = "/orange/ewhite/NeonData/2015_Campaign/D03/OSBS/L5/ITCs/"+ainput[:-4]+"_silva.shp"
ainput = sys.argv[1]
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
with rasterio.open(pt+ainput) as dataset:
    extent = dataset.transform
    data_crs = dataset.crs



from rasterio.transform import from_origin
final = final.astype(float)

pt_out = "/orange/ewhite/NeonData/2015_Campaign/D03/OSBS/L5/SPAnde/"	
new_dataset = rasterio.open(pt_out+ainput[:-4]+'_sp.tif', 'w', driver='GTiff',
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
	with rasterio.open(pt_out+ainput[:-4]+'_sp.tif') as src:
		affine = src.transform
		array = src.read(ii+1)
		df_zonal_stats = pd.DataFrame(zonal_stats(itc, array, affine=affine, stats=['mean']))   
	itc = pd.concat([itc, df_zonal_stats], axis=1)


poly_out = "/orange/ewhite/NeonData/2015_Campaign/D03/OSBS/L5/Sp_poly/"	
csv_name = poly_out+ainput[:-4]
itc.dropna().to_csv(csv_name+".csv", index=False) 







