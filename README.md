# Nighttime_TIR_georeferencing_improvement_ECOSTRESS
The programs use h5 files of ECOSTRESS, georeference them and use Google-Earth-Engine created masks as matching reference.

requires following packages: 
numpy
rasterio
pyplot
os
scipy
time
sys
copy
h5py
cv2
itertools
re
datetime
glob
gdal

In initial_georeferencing.py the paths need to be adapted. The program outputs GeoTiff images with name ending with '_georef.tif'. Three tif files are created: image file (LSTE), and annotation files to access quality flags from h5 file: _orig_px_coord_y- or x.tif 
The three files are further used in georeferencing improvement script.

In georeferencing_improvement.py, the variables that can/should be edited are placed up front. 
Please adapt paths to match your folder structure. The georeferenced files are expected to end with _georef.tif, change if needed. 
To create plots of matched water body edges, set the variable create_control_plots to True. 
The output of the script is a GeoTiff image and txt file with parameters: tie point locations and fitted transformation parameters.
It is possible to include scaling in the transformation, but the minimum number of tie points needs to be adapted. 
