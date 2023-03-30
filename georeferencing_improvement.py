import numpy as np
import rasterio
import matplotlib.pyplot as plt
import os
from glob import glob
from scipy import ndimage
from scipy.optimize import curve_fit
from time import time
import sys
import copy
import h5py
from scipy.ndimage.measurements import *
import cv2
from rasterio.warp import calculate_default_transform, reproject, Resampling
from itertools import combinations
import re
from matplotlib.colors import ListedColormap
from datetime import datetime

starttime = time()
path = '/data/geohot/ECOSTRESS_data/Georef/Olkaria/'
h5_file_path = '/data/geohot/ECOSTRESS_data/Raw_data/Olkaria/'
reference_path = '/data/geohot/SENTINEL2_SCL_BIG/'
os.chdir(path)
target_img_list = glob('*georef.tif')

minimum_LST = 250 ### Expected valid LST values in K (including clouds)
maximum_LST = 320 ### Expected valid LST values in K (including clouds)
threshold_sigma = 0.33 ### Canny edge parameter calculation. Lower value of sigma indicates a tighter threshold whereas a larger value gives a wider threshold.
minimum_lake_size = 50 ### Minimum size of a water body to be considered
test_range_span = 75 ### Half of the search window size (search window size = 2 * test_range_span)
minimum_edge_size = 10 ### ignore lakes with an edge that has less than 10 pixels
create_control_plots = False
minimum_num_tp = 2 ### Minimal number of tie points
rotation_span = 1.5 ### Upper bound of rotation
importance_evaluation_test_1 = 60 ### For lakes of this size, additional importance is given
importance_evaluation_test_2 = 200 ### For lakes of this size, additional importance is given
importance_evaluation_test_3 = 100 ### For lakes of this size, distribution test is conducted

minimum_validity = 0.15 ### Minimum validity for keeping a tie point
minimum_importance = 3 ### Minimum importance for keeping a tie point

def t_mat(point, offset_x, offset_y, angle):#, scale_x, scale_y):
    import numpy as np
    import sys
    angle_rad = angle / 180 * np.pi
    scale_x = 1
    scale_y = 1

    T = np.array([[1,0,offset_x],
                  [0,1,offset_y],
                  [0,0,1       ]])
    R = np.array([[np.cos(angle_rad), -np.sin(angle_rad),0],
                  [np.sin(angle_rad), np.cos(angle_rad),0],
                  [0,                 0,                1]])

    S = np.array([[scale_x,0,      0],
                  [0,      scale_y,0],
                  [0,      0,      1]])

    result = np.zeros([np.shape(point)[0],3])
    A = np.dot(np.dot(R,T),S)
    B = np.linalg.inv(A)

    for r in range(np.shape(point)[0]):
        result[r] = np.dot(A, np.hstack([point[r],1]))
        
    return(result[:,:2])

def read_qc(number):
    import numpy as np
    
    orig_img_qc = number

    ### Documented in ECOSTRESS_L2_ATBD_LSTE_2018-03-08.pdf

    ### Overall description of status of pixel
    bitmask_mandatory = 0x0003
    bitshift_mandatory = 0

    ### Data quality field
    bitmask_data_quality = 0x000C
    bitshift_data_quality = 2

    ### Cloud mask field, not set 
    bitmask_cloud = 0x0030 ### not set
    bitshift_cloud = 4      ### not set

    ### Number of iterations needed to remove reflected downwelling sky irradiance
    bitmask_iterations = 0x00C0
    bitshift_iterations = 6

    ### Atmospheric opacity test for humid scenes
    bitmask_atmospheric_opacity = 0x0300
    bitshift_atmospheric_opacity = 8

    ### MMD regime: MMD<0.3 (near-graybody) or MMD>0.3 (likely bare).
    bitmask_mmd = 0x0C00
    bitshift_mmd = 10

    ### Emissivity accuracy
    bitmask_emissivity = 0x3000
    bitshift_emissivity = 12

    ### LST accuracy
    bitmask_lst = 0xC000
    bitshift_lst = 14


    qc_flag_mandatory = ((orig_img_qc & bitmask_mandatory)>>bitshift_mandatory)
    ### 0 pixel produced, best quality, 1 pixel produced nominal quality, 2 pixel produced but cloud detected, 3 pixel not produced 

    qc_flag_data_quality = ((orig_img_qc & bitmask_data_quality)>>bitshift_data_quality)
    ### 0 good quality L1B data, 1 missing stripe pixel in bands 1 and 5, 2 not set, 3 missing or bad L1B data

    qc_flag_iterations = ((orig_img_qc & bitmask_iterations)>>bitshift_iterations)
    ### 0 slow convergence, 1 nominal , 2 nominal, 3 fast

    qc_flag_atmospheric_opacity = ((orig_img_qc & bitmask_atmospheric_opacity)>> bitshift_atmospheric_opacity)
    ### 0 warm humid air or cold land, 1 nominal value = 0.2 - 0.3, 2 nominal value = 0.1 - 0.2, 3 dry or high altitude pixel

    qc_flag_mmd = ((orig_img_qc & bitmask_mmd)>>bitshift_mmd)
    ### 0 most silicate rocks, 1 rocks, sand and some soils, 2 mostly soils, mixed pixel, 3 vegetation, snow, water, ice

    qc_flag_emissivity = ((orig_img_qc & bitmask_emissivity)>>bitmask_emissivity)
    ### 0 poor performance, 1 marginal performance, 2 good performance, 3 excellent performance

    qc_flag_lst = ((orig_img_qc & bitmask_lst)>>bitshift_lst)
    ### 0 poor performance, 1 marginal performance, 2 good performance, 3 excellent performance
    
    qc = [qc_flag_mandatory, qc_flag_data_quality, qc_flag_emissivity, qc_flag_lst]
    
    return(qc)

def euclidean_distance(p1,p2):
    import numpy as np
    
    p1 = np.array(p1)
    p2 = np.array(p2)
    
    if (p1.ndim == 1) & (p2.ndim == 1):
        euclidean = np.sqrt((p1-p2)[0]**2+(p1-p2)[1]**2)
    elif (p1.ndim == 2) & (p2.ndim ==2):
        p1_y = p1[:,0]
        p1_x = p1[:,1]
        
        p2_y = p2[:,0]
        p2_x = p2[:,1]
        
        
        y_distance = abs(p1_y - p2_y)
        x_distance = abs(p1_x - p2_x)
        
        euclidean = np.sqrt(y_distance**2+x_distance**2)
    return(euclidean)

def gaussian(v, mu, sigma, scale):
    import numpy as np

    gauss = np.zeros([len(v)])
    for x in range(len(v)):
        gauss[x] = (1/(sigma * np.sqrt(2*np.pi))) * np.exp(-0.5*((v[x]-mu)/sigma)**2)
    gauss = gauss * scale
    return(gauss)


for target_img_f in range(len(target_img_list)):
    ################ Data preparation ##################################
    part1_time = time()
    target_img_file_name = target_img_list[target_img_f]
    
    target_name_pattern = r'^ECOSTRESS_L2_LSTE_(?P<orbit_num>[0-9]{5})_(?P<scene_num>[0-9]{3})_(?P<target_acq_timestamp>[0-9]{8}T[0-9]{6}).*[.]tif'
    match = re.match(target_name_pattern, target_img_file_name)
    target_acq_timestamp = match.group('target_acq_timestamp')
    target_acq_year = target_acq_timestamp[:4]
    target_acq_month = int(target_acq_timestamp[4:6])
    
    target_h5_file = glob(h5_file_path + 'ECOSTRESS_L2_LSTE*'+ target_acq_timestamp + '*.h5')[0]
    
    ### 2D-Indices to Quality Flags 
    target_flag_file_x = glob('*' + target_acq_timestamp + '*px_coord_x.tif')[0]
    target_flag_file_y = glob('*' + target_acq_timestamp + '*px_coord_y.tif')[0]
    
    target_img_correct_name = target_img_file_name[:-4] + '_corr.tif'
    target_img_correct_orig_coords_x_name = target_flag_file_x[:-4] + '_corr.tif'
    target_img_correct_orig_coords_y_name = target_flag_file_y[:-4] + '_corr.tif'
    
    part1_1_time = time()
    
    ### open QC flag file, prepare quality/cloud mask
    y_new_coords_gtmax = h5py.File(target_h5_file, 'r')
    qc = np.array(y_new_coords_gtmax.get('SDS/QC'))
    qc_flags = read_qc(qc)
    qc_mandatory_flag = qc_flags[0]
    target_cloud_mask = copy.deepcopy(qc_mandatory_flag)
    target_cloud_mask[target_cloud_mask < 2] = 0
    target_cloud_mask[target_cloud_mask >= 2] = 1
    
    target_qc_idx_x = rasterio.open(target_flag_file_x).read(1).astype(int)
    target_qc_idx_y = rasterio.open(target_flag_file_y).read(1).astype(int)
    
    part1_2_time = time()
    
    ### Open ECOSTRESS target file
    target_img_file = rasterio.open(target_img_file_name)
    target_img_meta = target_img_file.meta
    target_img = target_img_file.read(1)
    target_imgK = target_img*0.02 ### convert DN to Kelvin
    target_bb_mask = np.zeros(target_img.shape) ### bb: bounding box
    target_bb_mask[target_img == 0] = 1
    
    ### threshold based cloud masking
    cloud_masking_bins = np.arange(minimum_LST,maximum_LST,1)
    LSThist = np.histogram(target_imgK.flatten(),bins=cloud_masking_bins)[0]
    LSThist_percent = LSThist/np.sum(LSThist)

    p0_cloud_masking = np.array([np.mean([minimum_LST, maximum_LST]),10,0.1])
    popt_cloud, pcov = curve_fit(gaussian, cloud_masking_bins[:-1], LSThist_percent, p0 = p0_cloud_masking)
    
    mu_cloud_masking, sigma_cloud_masking, scale_cloud_masking = popt_cloud
    threshold_cloud = mu_cloud_masking - (1.5 * sigma_cloud_masking)
    

    kernel_9x9 = np.ones((9,9), np.uint8)
    
    ### reduce usable image size (dilate the ones), in order to remove the outer borders in the derived edge image
    target_bb_mask_dilation = cv2.dilate(target_bb_mask, kernel_9x9, iterations = 1)
    
    target_cloud_mask_referenced = target_cloud_mask[target_qc_idx_y, target_qc_idx_x]
    target_cloud_mask_referenced_dilation = cv2.dilate(target_cloud_mask_referenced, kernel_9x9, iterations = 1)
    
    ### Open sentinel reference
    reference_orig_file_name = reference_path + 'olkaria-scl-water-' + target_acq_year + '.tif'
    reference_reproject_file_name = reference_path + 'olkaria-scl-water-' + target_acq_year + '_reproject.tif'
    
    reference_cloud_orig_file_name = reference_path + 'olkaria-scl-cloud-' + target_acq_year + '.tif'
    reference_cloud_reproject_file_name = reference_path + 'olkaria-scl-cloud' + target_acq_year + '_reproject.tif'
    
    try:
        reference_orig_file = rasterio.open(reference_orig_file_name)
        reference_orig = reference_orig_file.read(target_acq_month).astype(np.uint8)
        reference_orig_meta = reference_orig_file.meta
    except(IndexError):
        continue
    reference_cloud_orig_mask = rasterio.open(reference_cloud_orig_file_name).read(target_acq_month) 
    
    ### Reproject sentinel reference to match ECOSTRESS geoinformation
    with rasterio.open(reference_reproject_file_name, 'w', **target_img_meta) as dst:
        reproject(reference_orig, destination=rasterio.band(dst, 1),
            src_transform=reference_orig_file.transform,
            src_crs=reference_orig_file.crs,
            dst_transform=target_img_file.transform,
            dst_crs=target_img_file.crs,
            resampling=Resampling.nearest)
    
    with rasterio.open(reference_cloud_reproject_file_name, 'w', **target_img_meta) as dst:
        reproject(reference_cloud_orig_mask, destination=rasterio.band(dst, 1),
            src_transform=reference_orig_file.transform,
            src_crs=reference_orig_file.crs,
            dst_transform=target_img_file.transform,
            dst_crs=target_img_file.crs,
            resampling=Resampling.nearest)
    
    reference_reprojected_file = rasterio.open(reference_reproject_file_name)
    reference_reprojected = reference_reprojected_file.read(1)
    
    reference_reprojected_masked = copy.deepcopy(reference_reprojected)
    
    reference_cloud_reprojected = rasterio.open(reference_cloud_reproject_file_name).read(1)
    reference_cloud_reprojected_dilation = cv2.dilate(reference_cloud_reprojected, kernel_9x9, iterations = 1)
    
    ### Canny edge parameters for reference image
    edge_dilation_kernel_reference = np.array([[0,1,0],[1,1,1],[0,1,0]]).astype(np.uint8) 
    t_lower_reference = 0
    t_upper_reference = 0.5
    
    reference_reprojected_edge = cv2.Canny(reference_reprojected_masked.astype(np.uint8), t_lower_reference, t_upper_reference)
    reference_reprojected_edge_dilate = cv2.dilate(reference_reprojected_edge, edge_dilation_kernel_reference, iterations = 1)
    
    reference_reprojected_edge_dilate_masked = np.zeros(reference_reprojected_edge_dilate.shape)

    reference_reprojected_edge_dilate_masked[target_imgK < threshold_cloud] = 0
    reference_reprojected_edge_dilate_masked[target_cloud_mask_referenced_dilation == 1] = 0
    reference_reprojected_edge_dilate_masked[reference_cloud_reprojected_dilation == 1] = 0
    reference_reprojected_edge_dilate_masked[target_bb_mask_dilation == 1] = 0
    reference_reprojected_edge_dilate_masked[reference_reprojected_edge_dilate == 255] = 1
    
    ### Label the lakes in reference image
    reference_repojected_labeled, num_lakes_reference = label(reference_reprojected_masked)
    
    hist_reference_lakes, hist_reference_lakes_bin_edges = np.histogram(reference_repojected_labeled.flatten(), range(num_lakes_reference+1))
    lake_labels = np.flatnonzero(hist_reference_lakes[1:] > minimum_lake_size) + 1 ### lake_labels of large lakes

    ### Prepare ECOSTRESS target image, normalize for Canny Edge
    target_img_norm = copy.deepcopy(target_img)
    target_img_norm[target_img == 0] = np.nan
    target_img_norm[target_imgK < threshold_cloud] = np.nan
    target_img_norm[target_cloud_mask_referenced == 1] = np.nan
    
    target_img_norm = (target_img_norm - np.nanpercentile(target_img_norm,1))/(np.nanpercentile(target_img_norm,99) - np.nanpercentile(target_img_norm,1))*255
    target_img_norm[target_img_norm < 0] = 0
    target_img_norm[target_img_norm > 255] = 255
    
    target_img_norm = target_img_norm.astype(np.uint8)
    target_img_norm_masked = copy.deepcopy(target_img_norm).astype(float)
    target_img_norm_masked[target_img_norm == 0] = np.nan ### ignore bounding box and clouds, note high clipping values (255) shall stay
    
    ### Canny edge parameters
    t_lower_target = (1-threshold_sigma)*np.nanmedian(target_img_norm_masked)
    t_upper_target = (1+threshold_sigma)*np.nanmedian(target_img_norm_masked)
    
    edge_dilation_kernel_target = edge_dilation_kernel_reference
    
    target_img_blurred = cv2.GaussianBlur(target_img_norm,(5,5),0)
    target_img_edge = cv2.Canny(target_img_blurred, t_lower_target, t_upper_target)
    
    target_img_edge_dilate = cv2.dilate(target_img_edge, edge_dilation_kernel_target, iterations = 1)
    
    target_img_edge_masked = copy.deepcopy(target_img_edge_dilate)
    target_img_edge_masked[target_imgK < threshold_cloud] = 0
    target_img_edge_masked[target_cloud_mask_referenced_dilation == 1] = 0
    target_img_edge_masked[reference_cloud_reprojected_dilation == 1] = 0
    target_img_edge_masked[target_bb_mask_dilation == 1] = 0
    target_img_edge_masked[target_img_edge_masked == 255] = 1

    ####### Looking for matches ##################################
    tp_target_x = []
    tp_target_y = []
    tp_reference_x = []
    tp_reference_y = []
    importance_id = []
    lake_labels_used = []
    
    test_range_x = np.arange(-test_range_span,+test_range_span,1)
    test_range_y = np.arange(-test_range_span,+test_range_span,1)
    
    
    ### prepare list of label bounding boxes
    idc_sort_lake_labels = np.argsort(reference_repojected_labeled.ravel(), axis=None)
    sorted_lake_labels = reference_repojected_labeled.ravel()[idc_sort_lake_labels]
    start_of_label = np.where(np.diff(sorted_lake_labels))[0]+1
    start_of_label = np.append(np.array([0]), start_of_label)
    
    idx_label=0
    flat_idx_cur_label = idc_sort_lake_labels[start_of_label[idx_label]:start_of_label[idx_label+1]]
    flat_idx_cur_label = idc_sort_lake_labels[start_of_label[lake_labels[idx_label]]:start_of_label[lake_labels[idx_label]+1]]
    idx_y_cur_label, idx_x_cur_label = np.unravel_index(flat_idx_cur_label, np.shape(reference_repojected_labeled))
    reference_bb_x_lo = np.min(idx_x_cur_label)
    reference_bb_x_hi = np.max(idx_x_cur_label)
    reference_bb_y_lo = np.min(idx_y_cur_label)
    reference_bb_y_hi = np.max(idx_y_cur_label)
    
    for i in range(len(lake_labels)):
        sum_abs_diff_edges = np.zeros([len(test_range_y),len(test_range_x)])
        hist_sum_edges_val2 = np.zeros([len(test_range_y),len(test_range_x)])
        hist_sum_edges_val1 = np.zeros([len(test_range_y),len(test_range_x)])
        
        idx_label=i
        flat_idx_cur_label = idc_sort_lake_labels[start_of_label[lake_labels[idx_label]]:start_of_label[lake_labels[idx_label]+1]]
        idx_y_cur_label, idx_x_cur_label = np.unravel_index(flat_idx_cur_label, np.shape(reference_repojected_labeled))
        
        reference_bb_x_lo = np.min(idx_x_cur_label)
        reference_bb_x_hi = np.max(idx_x_cur_label)
        reference_bb_y_lo = np.min(idx_y_cur_label)
        reference_bb_y_hi = np.max(idx_y_cur_label)        
        
        reference_lake = reference_reprojected_edge_dilate_masked[reference_bb_y_lo:reference_bb_y_hi,reference_bb_x_lo:reference_bb_x_hi]
        if np.count_nonzero(reference_lake) < minimum_edge_size:
            continue

        x_idx_reference_lake = reference_bb_x_lo
        y_idx_reference_lake = reference_bb_y_lo
        
        for tx in range(len(test_range_x)):
            x_idx_target_test_position = x_idx_reference_lake + test_range_x[tx]
            ### Check if the current test position is not outside of the target image
            if (x_idx_target_test_position < 0): 
                continue
            if (x_idx_target_test_position + reference_lake.shape[1]) > target_img_edge_masked.shape[1]:
                continue
            for ty in range(len(test_range_y)):
                y_idx_target_test_position = y_idx_reference_lake + test_range_y[ty]
    
                if (y_idx_target_test_position < 0):
                    continue
                if (y_idx_target_test_position + reference_lake.shape[0]) > target_img_edge_masked.shape[0]:
                    continue
    
                target_lake = target_img_edge_masked[y_idx_target_test_position:y_idx_target_test_position+reference_lake.shape[0],x_idx_target_test_position:x_idx_target_test_position+reference_lake.shape[1]]
                
                sum_abs_diff_edges[ty,tx] = np.sum(abs(target_lake - reference_lake)) ### sum of (differences of edge images)

                ### Note: hist_sum_edges_val2 indicates     agreeing edges in target and reference
                hist_sum_edges_val2[ty,tx] = np.count_nonzero((target_lake + reference_lake).ravel()==2)
                ### Note: hist_sum_edges_val1 indicates dis-agreeing edges in target and reference
                hist_sum_edges_val1[ty,tx] = np.count_nonzero((target_lake + reference_lake).ravel()==1)

        ### No matches found whatsoever
        if np.nanmax(hist_sum_edges_val2) == 0:
            continue
        
        corr_position_target_lake_histtest_y, corr_position_target_lake_histtest_x = np.unravel_index(np.nanargmax(hist_sum_edges_val2),(len(test_range_y),len(test_range_x)))
        
        val2_val1_relation = hist_sum_edges_val2[corr_position_target_lake_histtest_y, corr_position_target_lake_histtest_x] / hist_sum_edges_val1[corr_position_target_lake_histtest_y, corr_position_target_lake_histtest_x]
        
        sum_abs_diff_edges[sum_abs_diff_edges == 0] = np.nan 
        
        ### Check if there are useful values in sum_abs_diff_edges
        if (np.sum(np.isnan(sum_abs_diff_edges)) == np.shape(sum_abs_diff_edges)[0]*np.shape(sum_abs_diff_edges)[1]) == False: 
            corr_position_target_lake_difftest_y, corr_position_target_lake_difftest_x = np.unravel_index(np.nanargmin(sum_abs_diff_edges),(len(test_range_y),len(test_range_x)))
        else:
            continue
        
        corr_target_lake_img_idx_y = y_idx_reference_lake + test_range_y[corr_position_target_lake_histtest_y] 
        corr_target_lake_img_idx_x = x_idx_reference_lake + test_range_x[corr_position_target_lake_histtest_x]
        
        lake_target_corr = target_img_edge_masked[corr_target_lake_img_idx_y:corr_target_lake_img_idx_y+reference_lake.shape[0],corr_target_lake_img_idx_x:corr_target_lake_img_idx_x+reference_lake.shape[1]]

        check_sum_reference_and_target_lake = lake_target_corr + reference_lake ### agreement map of reference and moved/corrected/shifted target reference_lake
        matching_pixels_y, matching_pixels_x = np.where(check_sum_reference_and_target_lake == 2)
        
        corr_target_lake_img_idx_mean_y = corr_target_lake_img_idx_y + np.mean(matching_pixels_y)
        corr_target_lake_img_idx_mean_x = corr_target_lake_img_idx_x + np.mean(matching_pixels_x)

        reference_lake_img_idx_mean_y = reference_bb_y_lo + np.mean(matching_pixels_y)
        reference_lake_img_idx_mean_x = reference_bb_x_lo + np.mean(matching_pixels_x)
        
        hist_0, hist_1, hist_2 = np.histogram(check_sum_reference_and_target_lake, bins = [0,1,2,3])[0]
        hist_0_target, hist_1_target = np.histogram(lake_target_corr, bins=[0,1,2])[0]
    
        match_validity = hist_2/(hist_1+hist_2)
        
        importance_index = 0
        
        if hist_2 > importance_evaluation_test_1: 
            importance_index += 1
            
        if hist_2 > importance_evaluation_test_2: 
            importance_index += 1
        
        if np.count_nonzero(reference_lake) > importance_evaluation_test_3: 
            ### check distribution of edges, shall be in all 4 quadrants
            check_sum_img_shape_y_half = int(check_sum_reference_and_target_lake.shape[0]/2)
            check_sum_img_shape_x_half = int(check_sum_reference_and_target_lake.shape[1]/2)
            
            #
            # Q1|Q2
            # Q3|Q4
            #
            
            test_results = 0
            
            check_sum_img_q1 = check_sum_reference_and_target_lake[:check_sum_img_shape_y_half,:check_sum_img_shape_x_half]
            check_sum_img_q2 = check_sum_reference_and_target_lake[:check_sum_img_shape_y_half,check_sum_img_shape_x_half:]
            check_sum_img_q3 = check_sum_reference_and_target_lake[check_sum_img_shape_y_half:,:check_sum_img_shape_x_half]
            check_sum_img_q4 = check_sum_reference_and_target_lake[check_sum_img_shape_y_half:,check_sum_img_shape_x_half:]
            
            if (2 in check_sum_img_q1):
                test_results += 1
            if (2 in check_sum_img_q2):
                test_results += 1
            if (2 in check_sum_img_q3):
                test_results += 1
            if (2 in check_sum_img_q4):
                test_results += 1        

            if test_results > 3:
                importance_index += 2
            elif test_results >= 2:
                importance_index += 1
    
        x_offset = x_idx_reference_lake - corr_target_lake_img_idx_x ### == - test_range_x[corr_position_target_lake_difftest_x]
        y_offset = y_idx_reference_lake - corr_target_lake_img_idx_y

        if (match_validity > minimum_validity) & (importance_index >= minimum_importance): 
            
            if create_control_plots == True:
                cm = ListedColormap(['#440154','#20908C','#FDE724'])
        
                fig, ax = plt.subplots(figsize = (6,4))
                plt.title('Label: %i,\n Validity: %.2f ' 'Importance: %i' %(labels[i], match_validity, importance_index))
                cax = ax.imshow(reference_lake + lake_target_corr, cmap = cm)
                cbar = fig.colorbar(cax, ticks = [0,1,2])
                cbar.ax.set_yticklabels(['background', 'no match', 'match']) 
                plt.savefig(target_img_file_name[:-4] + '_label_' + str(labels[i]) + '.png', bbox_inches='tight')
                plt.close()
            
            tp_target_x = np.append(tp_target_x, corr_target_lake_img_idx_x)
            tp_target_y = np.append(tp_target_y, corr_target_lake_img_idx_y)

            tp_reference_x = np.append(tp_reference_x, reference_bb_x_lo)
            tp_reference_y = np.append(tp_reference_y, reference_bb_y_lo)
            
            importance_id = np.append(importance_id, importance_index)
            lake_labels_used = np.append(lake_labels_used, lake_labels[i])
            
    ############ Find transformation matrix ###########################
    reference_coords = np.array([tp_reference_x, tp_reference_y]).T
    target_coords  = np.array([tp_target_x,  tp_target_y]).T
    
    tp_reference = reference_coords
    tp_target = target_coords
    
    ### Note: the variable tp_reference_subset will be evaluated at every function call, it can be changed after this definition here and new values will be reflected
    def t_mat2(tp_target, *p0):
        return (t_mat(tp_target, *p0)-tp_reference_subset).flatten() 
            
    def t_mat_flatten(tp_target, *p0):
        return t_mat(tp_target, *p0).flatten() 
            
    #### Look for outliers in transformation parameters
    tp_target_entries_idx = np.arange(len(tp_target)).astype(int) 
    tp_target_entries_idx_set = set(tp_target_entries_idx)

    ### Are enough tie points found?
    if ( len(tp_target_entries_idx) < minimum_num_tp ): 
        print('Too few lakes found')
        continue
    
    elif ( len(tp_target_entries_idx) == minimum_num_tp ):
        p0 = np.array([0,0,0])
        try:
            popt, pcov = curve_fit(t_mat_flatten, tp_target, tp_reference.flatten(), p0 = p0, bounds=((-test_range_span,-test_range_span,-rotation_span),(test_range_span,test_range_span,rotation_span)))
            tp_target_transformed = t_mat(tp_target, *popt)
            
            distance = euclidean_distance(tp_reference, tp_target_transformed)

        except(RuntimeError):
            continue
        
    elif ( len(tp_target_entries_idx) > minimum_num_tp ):
        combinations_array = np.array(list(combinations(tp_target_entries_idx_set, minimum_num_tp)))
        
        distance           = np.zeros([len(combinations_array),len(tp_target_entries_idx)])
        combination_labels           = np.zeros([len(combinations_array),len(tp_target_entries_idx)])
        popt_combinations  = np.zeros([len(combinations_array), 3])
        summary_importance = np.zeros([len(combinations_array)])
        
        for c in range(len(combinations_array)):
            labels_subset     = lake_labels_used[combinations_array[c]]
            
            tp_target_subset   = tp_target[combinations_array[c]]
            tp_reference_subset  = tp_reference[combinations_array[c]]
            importance_subset = importance_id[combinations_array[c]]
            
            p0 = np.array([0,0,0])
            try:
                popt, pcov = curve_fit(t_mat_flatten, tp_target_subset, tp_reference_subset.flatten(), p0 = p0, bounds=((-test_range_span,-test_range_span,-rotation_span),(test_range_span,test_range_span,rotation_span)))
            except(RuntimeError):
                continue
            popt_combinations[c] = popt
    
            tp_target_transformed = t_mat(tp_target, *popt)
            
            distance[c] = euclidean_distance(tp_reference, tp_target_transformed)
 
        if np.sum(np.isnan(distance)) == np.size(distance):
            print('Only nans found')
            continue
    
        eucl_dist_threshold_small = 3

        tp_target_number_of_small_eucl_dist = np.nansum(distance <= eucl_dist_threshold_small,axis=0)
        combination_number_of_small_eucl_dist = np.nansum(distance <= eucl_dist_threshold_small,axis=1)
    
        inlier_idx = np.unique(combinations_array[np.where(combination_number_of_small_eucl_dist >= np.percentile(combination_number_of_small_eucl_dist,80))])
        
        tp_reference_subset_final = tp_reference[inlier_idx]
        tp_target_subset_final = tp_target[inlier_idx]
    
        if np.size(tp_target_subset_final) < 2:
            print('tp_target_subset_final is too small')
            continue
        
        p0 = np.array([0,0,0])
        try:
            popt, pcov = curve_fit(t_mat_flatten, tp_target_subset_final, tp_reference_subset_final.flatten(), p0 = p0, bounds=((-test_range_span,-test_range_span,-rotation_span),(test_range_span,test_range_span,rotation_span)))
    
        except(RuntimeError):
            print('Optimal parameters not found', target_img_file_name)
        
        tp_target_transformed_final = t_mat(tp_target_subset_final, *popt)
        tp_distance_final = euclidean_distance(tp_reference_subset_final, tp_target_transformed_final)
        
        tp_reference_subset_final_corrected = np.delete(tp_reference_subset_final, np.where(tp_distance_final > eucl_dist_threshold_small), axis = 0)
        tp_target_subset_final_corrected = np.delete(tp_target_subset_final, np.where(tp_distance_final > eucl_dist_threshold_small), axis = 0)
        if len(tp_reference_subset_final_corrected) > 1:    
            try:
                popt, pcov = curve_fit(t_mat_flatten, tp_target_subset_final_corrected, tp_reference_subset_final_corrected.flatten(), p0 = p0, bounds=((-test_range_span,-test_range_span,-rotation_span),(test_range_span,test_range_span,rotation_span)))
                tp_target_transformed_final_corrected = t_mat(tp_target_subset_final_corrected, *popt)
                tp_distance_final = euclidean_distance(tp_reference_subset_final_corrected, tp_target_transformed_final_corrected)
                tp_reference_subset_final = tp_reference_subset_final_corrected
                tp_target_subset_final = tp_target_subset_final_corrected
            except(RuntimeError):
                print('Optimal parameters not found', target_img_file_name)
        else:
            tp_distance_final = euclidean_distance(tp_reference_subset_final, tp_target_transformed_final)

        part4_time = time()
    
    ################################## Image resampling ##################################
    img_target_correct               = np.zeros([reference_reprojected_edge_dilate_masked.shape[0], reference_reprojected_edge_dilate_masked.shape[1]])
    img_target_correct_qc_coords_x = np.zeros([reference_reprojected_edge_dilate_masked.shape[0], reference_reprojected_edge_dilate_masked.shape[1]])
    img_target_correct_qc_coords_y = np.zeros([reference_reprojected_edge_dilate_masked.shape[0], reference_reprojected_edge_dilate_masked.shape[1]])
    img_target_correct_edge          = np.zeros([reference_reprojected_edge_dilate_masked.shape[0], reference_reprojected_edge_dilate_masked.shape[1]])
    
    start_resampling = time()
    
    resampled_target_img_x_array = np.arange(0,np.shape(img_target_correct)[1],1)
    resampled_target_img_y_array = np.arange(0,np.shape(img_target_correct)[0],1)
    
    resampled_target_img_x_mesh, resampled_target_img_y_mesh = np.meshgrid(resampled_target_img_x_array, resampled_target_img_y_array)
    
    resampled_target_img_x_mesh_flat = resampled_target_img_x_mesh.flatten()
    resampled_target_img_y_mesh_flat = resampled_target_img_y_mesh.flatten()

    coords = np.array([resampled_target_img_x_mesh_flat, resampled_target_img_y_mesh_flat]).T
    
    old_coords = np.round(t_mat_inv(coords, *popt)).astype(int)
    
    max_y_coord = np.min([np.shape(target_img)[0], np.shape(img_target_correct)[0]]).astype(int)-1
    max_x_coord = np.min([np.shape(target_img)[1], np.shape(img_target_correct)[1]]).astype(int)-1
    
    y_old_coords_lt0 = []
    y_old_coords_lt0 = np.append(y_old_coords_lt0, np.where(old_coords < 0)[0])
    y_old_coords_lt0_unique = np.unique(y_old_coords_lt0)
    
    y_new_coords_lt0 = []
    y_new_coords_lt0 = np.append(y_new_coords_lt0, np.where(coords < 0)[0])
    y_new_coords_lt0_unique = np.unique(y_new_coords_lt0)
    
    x_old_coords_gtmax = []
    x_old_coords_gtmax = np.append(x_old_coords_gtmax, np.where(old_coords[:,1] > max_y_coord)[0])
    x_old_coords_gtmax_unique = np.unique(x_old_coords_gtmax)
    
    x_new_coords_gtmax = []
    x_new_coords_gtmax = np.append(x_new_coords_gtmax, np.where(coords[:,1] > max_y_coord)[0])
    x_new_coords_gtmax_unique = np.unique(x_new_coords_gtmax)
    
    y_old_coords_gtmax = []
    y_old_coords_gtmax = np.append(y_old_coords_gtmax, np.where(old_coords[:,0] > max_x_coord)[0])
    y_old_coords_gtmax_unique = np.unique(y_old_coords_gtmax)
    
    y_new_coords_gtmax = []
    y_new_coords_gtmax = np.append(y_new_coords_gtmax, np.where(coords[:,0] > max_x_coord)[0])
    y_new_coords_gtmax_unique = np.unique(y_new_coords_gtmax)
    
    incorrect_indices = np.concatenate([y_old_coords_lt0_unique, y_new_coords_lt0_unique, x_old_coords_gtmax_unique, x_new_coords_gtmax_unique, y_old_coords_gtmax_unique, y_new_coords_gtmax_unique]).astype(int)
    
    old_coords_drop = np.delete(old_coords, incorrect_indices, axis = 0).astype(int)
    coords_drop = np.delete(coords, incorrect_indices, axis = 0).astype(int)
    
    img_target_correct[coords_drop[:,1], coords_drop[:,0]] = target_img[old_coords_drop[:,1], old_coords_drop[:,0]]
    
    img_target_correct_qc_coords_x[coords_drop[:,1], coords_drop[:,0]] = target_qc_idx_x[old_coords_drop[:,1], old_coords_drop[:,0]]
    
    img_target_correct_qc_coords_y[coords_drop[:,1], coords_drop[:,0]] = target_qc_idx_y[old_coords_drop[:,1], old_coords_drop[:,0]]
    
    img_target_correct_reshape = np.reshape(img_target_correct,[1,img_target_correct.shape[0],img_target_correct.shape[1]]).astype('float32')
    
    img_target_correct_orig_coords_x_reshape = np.reshape(img_target_correct_qc_coords_x,[1,img_target_correct.shape[0],img_target_correct.shape[1]]).astype('float32')
    
    img_target_correct_orig_coords_y_reshape = np.reshape(img_target_correct_qc_coords_y,[1,img_target_correct.shape[0],img_target_correct.shape[1]]).astype('float32')
    
    with rasterio.open(save_path + target_img_correct_name, "w", **target_img_meta) as dest:
        dest.write(img_target_correct_reshape)
    with rasterio.open(save_path + target_img_correct_orig_coords_x_name, "w", **target_img_meta) as dest:
        dest.write(img_target_correct_orig_coords_x_reshape)
    with rasterio.open(save_path + target_img_correct_orig_coords_y_name, "w", **target_img_meta) as dest:
        dest.write(img_target_correct_orig_coords_y_reshape)
    
    params_file = open(save_path + target_img_file_name[:-4] + '_params.txt', 'w') 
    params_file.write('File saved ' + str(datetime.now()))
    params_file.write('Fitted parameters (x,y,rot): ' + str(popt) + '\n')
    params_file.write('reference tp locations [x,y]' + str(tp_reference_subset_final).replace('\n', ',') + '\n')
    params_file.write('target tp locations [x,y]' + str(tp_target_subset_final).replace('\n', ',') + '\n')
    params_file.write('Euclidean distance [px]' + str(tp_distance_final) + '\n')
    params_file.close()
    print('saved ' + target_img_correct_name)
