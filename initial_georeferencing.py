import numpy as np
import h5py
import os
from time import time
from osgeo import gdal, osr
from glob import glob
import re

def deg2m(lat_input):
    import math 

    deg2rad = math.pi / 180 # Degrees to radians variable
    
    lat_rad = lat_input*deg2rad # Latitude degrees into radians
    meters_per_degree_lat = 111132.92 - 559.82 * math.cos(2*lat_rad) + 1.175 * math.cos(4*lat_rad) #Calculate meters pro degree latitude
    meters_per_degree_lon =          111412.84 * math.cos(lat_rad)    - 93.5 * math.cos(3*lat_rad) #Calculate meters pro degree longitude
    
    return(meters_per_degree_lat, meters_per_degree_lon)

def array2raster(newRasterfn,rasterOrigin,pixelWidth,pixelHeight,array):

    cols = array.shape[1]
    rows = array.shape[0]
    originX = rasterOrigin[0]
    originY = rasterOrigin[1]

    driver = gdal.GetDriverByName('GTiff')
    outRaster = driver.Create(newRasterfn, cols, rows, 1, gdal.GDT_Float32)
    outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, -pixelHeight))
    outband = outRaster.GetRasterBand(1)
    outband.WriteArray(array)
    outRasterSRS = osr.SpatialReference()
    outRasterSRS.ImportFromEPSG(4326)
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband.FlushCache()
    outRaster = None

path = '/mnt/hgfs/PostDoc/ECOSTRESS_data/Raw_download/Raw_data/'
save_path = '/mnt/hgfs/PostDoc/ECOSTRESS_data/Raw_download/Georef/'
os.chdir(path)
lste_list = glob('ECOSTRESS_L2_LSTE*.h5')
lste_list_to_do = []

os.chdir(save_path)
already_georeferenced_file_list = glob('*georef.tif')
already_georeferenced_file_names = []

for a in range(len(already_georeferenced_file_list)):
    already_georeferenced_file_names = np.append(already_georeferenced_file_names, already_georeferenced_file_list[a][:51])

for l in range(len(lste_list)):
    lste_name = lste_list[l][:-3]
    
    if (lste_name in already_georeferenced_file_names) == False:
        lste_list_to_do = np.append(lste_list_to_do, lste_list[l])

os.chdir(path)
for target_img_file_name in lste_list_to_do:
    print('Working on: ', target_img_file_name)
    starttime = time()
    target_name_pattern = r'^ECOSTRESS_L2_LSTE_(?P<orbit_num>[0-9]{5})_(?P<scene_num>[0-9]{3})_(?P<target_acq_timestamp>[0-9]{8}T[0-9]{6}).*[.]h5'
    match = re.match(target_name_pattern, target_img_file_name)
    target_acq_timestamp = match.group('target_acq_timestamp')
    
    geo_file_list = glob('*GEO_*' + target_acq_timestamp + '*.h5')
    sys.exit()
    if len(geo_file_list) != 0:
        geo_file = geo_file_list[0]
    else:
        print('Error: no GEO file found')
        continue

    ### set name for the georeferenced image
    georef_name = target_img_file_name[:-3] + '_georef'

    ### load image file
    try:
        f = h5py.File(target_img_file_name, 'r')
    except(OSError):
        print('Error: ', target_img_file_name)
        continue
    data = f.get('SDS/LST')
    qc = f.get('SDS/QC')
    orig_img_lst = np.array(data)
    orig_img_qc = np.array(qc)

    ### find the dimensions of the original image
    orig_img_size_y = np.shape(orig_img_lst)[0] # direction of latitudes
    orig_img_size_x = np.shape(orig_img_lst)[1] # direction of longitudes


    ### load the geo file and derive two arrays: for latitude and for longitude
    try:
        g = h5py.File(geo_file, 'r')
    except:
        print('Error: ',georef_name)
        continue
    ls = list(g.keys())
    geo_obj = []
    g.visit(geo_obj.append)
    geo_lat = g.get('Geolocation/latitude')
    geo_lon = g.get('Geolocation/longitude')
    orig_img_lat_deg = np.array(geo_lat)
    orig_img_lon_deg = np.array(geo_lon)

    ### calculate how many meters per degree latitude do we have in this case
    deg2m_lat = deg2m(np.mean(orig_img_lat_deg))[0]
    deg2m_lon = deg2m(np.mean(orig_img_lat_deg))[1]

    ### recalculate the lat und lon arrays into meters
    orig_img_lat_m = orig_img_lat_deg*deg2m_lat
    orig_img_lon_m = orig_img_lon_deg*deg2m_lon

    ### one resampled pixel has a foot print on ground of the following amount of metres
    ### ECOv002_L2-4_UserGuide: All of the raster layers in all of the L2G/L3G/L4G gridded products are projected to a globally snapped 0.0006° grid in WGS84 latitude and longitude to approximate 70 m resolution
    resampled_px_size_lat_m = deg2m_lat * 0.0006
    resampled_px_size_lon_m = deg2m_lon * 0.0006

    #find out a bounding box
    BB_W = 0
    BB_S = 1
    BB_E = 2
    BB_N = 3
    
    ### dimensions of the bounding box in degrees and meters
    resampled_bb_dims_deg = np.array([np.amin(orig_img_lon_deg)          ,  np.amin(orig_img_lat_deg)         , np.amax(orig_img_lon_deg)          , np.amax(orig_img_lat_deg)          ])
    resampled_bb_dims_m   = np.array([np.amin(orig_img_lon_m)            , np.amin(orig_img_lat_m)            , np.amax(orig_img_lon_m)            , np.amax(orig_img_lat_m)            ])
    ### set templates for lat and lon grid of the resampled image in metres
    ### size of the georeferenced image in y and x directions
    resampled_img_size_y = int(np.ceil((resampled_bb_dims_m[BB_N]-resampled_bb_dims_m[BB_S])/resampled_px_size_lat_m)) # direction of latitudes
    resampled_img_size_x = int(np.ceil((resampled_bb_dims_m[BB_E]-resampled_bb_dims_m[BB_W])/resampled_px_size_lon_m)) # direction of longitude
    ###                                          number of pixels      * meters per degree       + meters on the edge of the image (south or west)
    resampled_lat_template_m = (np.matrix(range(resampled_img_size_y)) * resampled_px_size_lat_m + resampled_bb_dims_m[BB_S]).T
    resampled_lon_template_m = (np.matrix(range(resampled_img_size_x)) * resampled_px_size_lon_m + resampled_bb_dims_m[BB_W])

    ### build the lat and lon grid of the resampled image
    resampled_lat_whole_m = np.matrix(np.zeros([resampled_img_size_y, resampled_img_size_x]))
    for i in range(resampled_img_size_x):
        resampled_lat_whole_m[:,i] = resampled_lat_template_m

    resampled_lon_whole_m = np.matrix(np.zeros([resampled_img_size_y, resampled_img_size_x]))
    for i in range(resampled_img_size_y):
        resampled_lon_whole_m[i,:] = resampled_lon_template_m

    ### create the glts for y(lat) and x(lon)
    resampled_glt_y = np.ones([resampled_img_size_y, resampled_img_size_x], dtype='int')*(-1)
    resampled_glt_x = np.ones([resampled_img_size_y, resampled_img_size_x], dtype='int')*(-1)

    ### the following block of code shall replace the loop version, as it is dramatically faster
    ###             round down  meters per latitude in original img
    resampled_idx_y = np.floor((orig_img_lat_m - resampled_bb_dims_m[BB_S]) / resampled_px_size_lat_m).astype('int')
    resampled_idx_x = np.floor((orig_img_lon_m - resampled_bb_dims_m[BB_W]) / resampled_px_size_lon_m).astype('int')
    (orig_YY, orig_XX) = np.meshgrid(range(orig_img_size_y), range(orig_img_size_x), indexing = 'ij') ### one could also use np.indices
    resampled_glt_y[resampled_idx_y,resampled_idx_x] = orig_YY
    resampled_glt_x[resampled_idx_y,resampled_idx_x] = orig_XX

    ### create the resampled and georegistered image
    resampled_img_lst = np.zeros([resampled_img_size_y, resampled_img_size_x])
    orig_px_y_coord   = np.zeros([resampled_img_size_y, resampled_img_size_x])
    orig_px_x_coord   = np.zeros([resampled_img_size_y, resampled_img_size_x])

    ### window size to look around the pixel indicated by the GLT
    ws_geo_y = int(10) 
    ws_geo_x = int(10) 

    process = np.zeros([resampled_img_size_y, resampled_img_size_x], dtype = 'bool')
    test_img = np.zeros([resampled_img_size_y, resampled_img_size_x])
    for resampled_img_idx_y in range(resampled_img_size_y-1):
        for resampled_img_idx_x in range(resampled_img_size_x-1):
            
            ### calculate lat and lon coordinates
            cur_lat = resampled_bb_dims_deg[BB_S] + resampled_img_idx_y * resampled_px_size_lat_m/deg2m_lat
            cur_lon = resampled_bb_dims_deg[BB_W] + resampled_img_idx_x * resampled_px_size_lon_m/deg2m_lon
            
            ### compare lat and lon with the original image
            orig_idx_approx_y = resampled_glt_y[resampled_img_idx_y,resampled_img_idx_x]
            orig_idx_approx_x = resampled_glt_x[resampled_img_idx_y,resampled_img_idx_x]

            resampled_img_idx_y_new = resampled_img_idx_y
            resampled_img_idx_x_new = resampled_img_idx_x
            if (resampled_glt_y[resampled_img_idx_y_new,resampled_img_idx_x_new] == -1 or resampled_glt_x[resampled_img_idx_y_new,resampled_img_idx_x_new] == -1):
                if   (resampled_img_idx_y_new -1 >= 0):
                    if(resampled_glt_y[resampled_img_idx_y_new-1, resampled_img_idx_x_new] != -1):
                        resampled_img_idx_y_new = resampled_img_idx_y_new -1

                if (resampled_img_idx_y_new +1 <= resampled_glt_y.shape[0]-1):
                    if (resampled_glt_y[resampled_img_idx_y_new+1, resampled_img_idx_x_new] != -1):
                        resampled_img_idx_y_new = resampled_img_idx_y_new +1

                if (resampled_img_idx_x_new -1 >= 0):
                    if (resampled_glt_x[resampled_img_idx_y_new, resampled_img_idx_x_new-1] != -1):
                        resampled_img_idx_x_new = resampled_img_idx_x_new -1

                if (resampled_img_idx_x_new +1 <= resampled_glt_x.shape[1]-1):
                    if (resampled_glt_x[resampled_img_idx_y_new, resampled_img_idx_x_new+1] != -1):
                        resampled_img_idx_x_new = resampled_img_idx_x_new +1
                
                if (resampled_glt_y[resampled_img_idx_y_new,resampled_img_idx_x_new] == -1):
                    continue

            orig_idx_approx_y = resampled_glt_y[resampled_img_idx_y_new,resampled_img_idx_x_new]
            orig_idx_approx_x = resampled_glt_x[resampled_img_idx_y_new,resampled_img_idx_x_new]
            
            y_lo = orig_idx_approx_y - ws_geo_y
            y_hi = orig_idx_approx_y + ws_geo_y +1
            x_lo = orig_idx_approx_x - ws_geo_x
            x_hi = orig_idx_approx_x + ws_geo_x +1
            ### boundary checks
            if (y_lo < 0):
                y_lo = 0
            if (x_lo < 0):
                x_lo = 0
            if (y_hi >= orig_img_size_y):
                y_hi = orig_img_size_y-1
            if (x_hi >= orig_img_size_x):
                x_hi = orig_img_size_x-1
            orig_img_lat_window = orig_img_lat_deg[y_lo:y_hi, x_lo:x_hi]
            orig_img_lon_window = orig_img_lon_deg[y_lo:y_hi, x_lo:x_hi]
            
            diff_lat_lon = np.sqrt( (orig_img_lat_window - cur_lat)**2 + (orig_img_lon_window - cur_lon)**2 )
            
            (orig_img_idx_y_nearest_win, orig_img_idx_x_nearest_win) = np.unravel_index( np.argsort( diff_lat_lon, axis = None  ) , np.shape(diff_lat_lon) )
            
            img_idx_y_first  = y_lo + orig_img_idx_y_nearest_win[0]
            img_idx_x_first  = x_lo + orig_img_idx_x_nearest_win[0]
            
            
            resampled_img_lst[resampled_img_idx_y,resampled_img_idx_x] = orig_img_lst[img_idx_y_first, img_idx_x_first]
                
            orig_px_y_coord[resampled_img_idx_y,resampled_img_idx_x] = img_idx_y_first
            orig_px_x_coord[resampled_img_idx_y,resampled_img_idx_x] = img_idx_x_first
                

    ### IR cameras write from down to up
    resampled_img_lst = np.flipud(resampled_img_lst)

    ### Bounding box[[min_lat, min_lon], [max_lat, max_lon]]
    [lat_max, lon_min, lat_min, lon_max] = [np.max(orig_img_lat_deg), np.min(orig_img_lon_deg), np.min(orig_img_lat_deg), np.max(orig_img_lon_deg)]
    resampled_px_size_lon = resampled_px_size_lon_m / deg2m_lon
    resampled_px_size_lat = resampled_px_size_lat_m / deg2m_lat

    pixelWidth  = resampled_px_size_lon 
    pixelHeight = resampled_px_size_lat

    origin_e = lon_min 
    origin_n = lat_max 
    rasterOrigin = [origin_e, origin_n]
    # row_rotation = 0
    # col_rotation = 0

    array2raster(save_path+georef_name + '.tif',rasterOrigin,pixelWidth,pixelHeight,resampled_img_lst)
    array2raster(save_path+georef_name + '_orig_px_coord_y.tif',rasterOrigin,pixelWidth,pixelHeight,orig_px_y_coord)
    array2raster(save_path+georef_name + '_orig_px_coord_x.tif',rasterOrigin,pixelWidth,pixelHeight,orig_px_x_coord)
    print('Saved dataset: ', georef_name, str(time() - starttime))