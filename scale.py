import sys

import numpy as np
import pandas as pd

import ccbid
from ccbid import args
from ccbid import prnt



pt = "/orange/ewhite/NeonData/2015_Campaign/D03/OSBS/L5/Rasters/"
#ainput = "NIS1_20140507_143910_atmcor_BRTYP.tif"
ainput = sys.argv[1]
aitcput = "/orange/ewhite/NeonData/2015_Campaign/D03/OSBS/L5/ITCs/"+ainput[:-4]+"_silva.shp"
aremove_outliers = 'PCA'
athreshold = 3
aaggregate = 'average'
# args.feature_selection = False

# set the seed for reproducibility (to the year the CCB was founded)
np.random.seed(1984)
model = ccbid.read.pck("./model/ensemble")

# get base data from the model
sp_labels = model.labels_

from skimage import io
features = np.array(io.imread(pt+ainput))
npix = features.shape
features = features.flatten().reshape(npix[0]*npix[1], 369) /10000

ndvi = (features[:,90] - features[:, 58])/(features[:,58] +features[:,90]) <0.5
nir860 = (features[:,96] + features[:,97])/2 < 0.4
naval = ~(ndvi | nir860)
mask1 = np.all(features < 0, axis=1) | np.all(features > 1, axis=1) | naval
data = features#[~mask1] 
#np.apply_along_axis(my_func, 0, b)
normMat = np.apply_along_axis(np.sum, 1, data**2)
normMat = np.sqrt(normMat)
normMat = np.tile(normMat, (data.shape[1],1))

data=data / np.transpose(normMat)
normMat = None
data.shape
#normMat=df.drop(['crown_id'], axis=1) / np.transpose(normMat)
mask_pca = ccbid.outliers.with_pca(data, thresh=athreshold)
data = data[mask_pca, 24:369]


data = model.reducer.transform(data)
data = data[:, 0:model.n_features_]
prnt.status("Applying CCBID model to input features")
prob = model.predict_proba(data, average_proba=True)

output = pd.DataFrame(index=range(mask_pca.shape[0]), columns=range(prob.shape[1]))
output.loc[mask_pca,:] =  prob
final = pd.DataFrame(index=range(features.shape[0]), columns=range(prob.shape[1]))
final[~mask1] = output
final = final.values.reshape(npix[0], npix[1], prob.shape[1])
final = final.astype(float)


import geopandas as gpd
import rasterio
from rasterio import features
from rasterio.transform import from_origin
from rasterstats import zonal_stats

with rasterio.open(pt+ainput) as dataset:
    extent = dataset.transform
    data_crs = dataset.crs


#get the crown ids from the crown delinetation
itc = gpd.read_file(aitcput)
itc.crs = data_crs

pt_out = "/orange/ewhite/NeonData/2015_Campaign/D03/OSBS/L5/SPAnde/"    
fname = pt_out+ainput[:-4]+'_sp.tif'
new_dataset = rasterio.open(fname, 'w', driver='GTiff',
                            height = final.shape[0], width = final.shape[1],
                            count=prob.shape[1], dtype=str(final.dtype),
                            transform=extent,
                            crs=data_crs)

for ii in range(prob.shape[1]):
    tmp = final[:,:,ii].astype(float)
    new_dataset.write(tmp, ii+1)


new_dataset.close()


for ii in range(prob.shape[1]):
    with rasterio.open(pt_out+ainput[:-4]+'_sp.tif') as src:
        affine = src.transform
        array = src.read(ii+1)
        df_zonal_stats = pd.DataFrame(zonal_stats(itc, array, affine=affine, stats=['mean']))
    # adding statistics back to original GeoDataFrame
    itc = pd.concat([itc, df_zonal_stats], axis=1) 


poly_out = "/orange/ewhite/NeonData/2015_Campaign/D03/OSBS/L5/Sp_poly/" 
csv_name = poly_out+ainput[:-4]
#itc.to_file(csv_name+".shp")

itc.dropna().to_csv(csv_name+".csv", index=False) 











