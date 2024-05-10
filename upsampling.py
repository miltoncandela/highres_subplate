import numpy as np
import nibabel as nib
import os
# import cv2
from scipy.ndimage import gaussian_filter, binary_erosion

core_path = '/neuro/labs/grantlab/research/MRI_processing/milton.candela/highres_subplate'
env_path = '/neuro/labs/grantlab/research/MRI_processing/milton.candela/envs/sp_env/bin/'

def smooth(subject, sigma, tau):
    """
    Smooth the borders of segmentation for binary regions 160 and 161.
    Bivariate Gaussian Filter (sigma, tau) for inner and outer SP

    Parameters:
        input_file (str): Path to the input .nii file.
        output_file (str): Path to save the output .nii file.
        sigma (float): Standard deviation for Gaussian blur (inner).
        tau (float): Standard deviation for Gaussian blur (outer).
    """
    
    # Load the image using nibabel
    nii_img = nib.load('{}/{}/{}_hs_sp_seg_upsamp_al.nii'.format(subjects_folder, subject, subject))
    data = nii_img.get_fdata(dtype=np.float32)

    regions_dict = {'iz': [160, 161], 'sp': [4, 5], 'cp': [42, 1], 'back': 0}
    
    def smooth_label(data, label, sigma):        
        # Extract binary regions
        region = (data == label).astype(np.float32)
        
        # Apply Gaussian blur to the binary masks
        blurred_region = gaussian_filter(region, sigma=sigma)
        
        # Determine new threshold based on the maximum value after blurring
        threshold = blurred_region.max() * 0.5
        
        # Threshold the blurred images to get binary results
        smoothed_region = (blurred_region >= threshold).astype(np.uint8)
    	
    	# Combine the smoothed masks back with the original data
    	# Ensure the original region data is preserved in areas not smoothed
        smoothed_data = np.where(smoothed_region == 1, label, data)
    	
        return smoothed_data
    
    # Applies the Sequential Gaussian Multilayer Smoothing (SGMS)
    smoothed_iz = smooth_label(smooth_label(data, regions_dict['iz'][0], sigma), regions_dict['iz'][1], sigma)
    merged_data = np.where((smoothed_iz != regions_dict['iz'][0]) & (smoothed_iz != regions_dict['iz'][1]), 0, smoothed_iz)
    smoothed_back = smooth_label(merged_data, regions_dict['back'], tau)

    #for pasted_regions in regions_dict['sp'] + regions_dict['cp']:
    #    smoothed_back = np.where((smoothed_iz == pasted_regions).astype(np.float32) == 1, pasted_regions, smoothed_back)
    
    # Save the smoothed image
    output_img = nib.Nifti1Image(smoothed_back, nii_img.affine, nii_img.header)
    nib.save(output_img, '{}/{}/{}_hs_sp_seg_smooth.nii'.format(subjects_folder, subject, subject))
def paste(subject):
    cp_img = nib.load('{}/{}/{}_hs_cp_seg.nii'.format(subjects_folder, subject, subject))
    data_cp = cp_img.get_fdata(dtype=np.float32)

    sp_img = nib.load('{}/{}/{}_hs_sp_seg_smooth.nii'.format(subjects_folder, subject, subject))
    data_sp = sp_img.get_fdata(dtype=np.float32)
    data_sp = np.squeeze(data_sp)

    # Flipping Detection Algorithm (FDA)
    def fda(data_cp, data_sp):
        """
        :return: bool, whether to flip or not to flip
        """

        izL_mask_sp = (data_sp == 160).astype(np.uint8)
        izL_mask_cp = (data_cp == 160).astype(np.uint8)

        # Calculate the Jaccard coefficient
        intersection = np.logical_and(izL_mask_sp, izL_mask_cp)
        union = np.logical_or(izL_mask_sp, izL_mask_cp)
        jaccard = np.sum(intersection) / np.sum(union)

        if jaccard < 0.2:
            return True
        else:
            return False

    if fda(data_cp, data_sp):
        data_sp = np.flip(data_sp, axis=0)
    

    def transform_mask(mask, oper):
        # Define the dilation kernel (3x3x3)
        kernel = np.array([[[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                        [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                        [[1, 1, 1], [1, 1, 1], [1, 1, 1]]], dtype=np.uint8)
        padded_mask = np.pad(mask, ((1, 1), (1, 1), (1, 1)), mode='constant')  # Pad the mask
        transformed_mask = np.zeros_like(mask)  # Initialize the dilated mask
        
        # Perform dilation
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                for k in range(mask.shape[2]):
                    if oper == 'dilate':
                        if np.any(padded_mask[i:i+3, j:j+3, k:k+3] & kernel):
                            transformed_mask[i, j, k] = 1
                    elif oper == 'erode':
                        if not np.all(padded_mask[i:i+3, j:j+3, k:k+3] & kernel == kernel):
                            transformed_mask[i, j, k] = 0
                    
        return transformed_mask

    regions_dict = {'left': [4, 160, 42], 'right': [5, 161, 1]}
    # Plan #
    # CP: 161 -> 5
    # Paste SP 161 to CP

    def paste_data(data, region):
        sp_label, iz_label, cp_label = regions_dict[region]

        data = np.where(data == iz_label, sp_label, data)
        iz_mask = (data_sp == iz_label).astype(np.uint8)

        cp_cond_mask = np.bitwise_and(iz_mask, (data_cp == cp_label).astype(np.uint8))
        no_cond_mask = np.bitwise_and(iz_mask, (data_cp == 0).astype(np.uint8))
        data = np.where((iz_mask == 1) & (cp_cond_mask != 1) & (no_cond_mask != 1), iz_label, data)

        # iz_mask = (data == iz_label).astype(np.uint8)
        # er_mask = transform_mask(iz_mask, 'erode')
        # er_cp_mask = np.bitwise_and(er_mask, (data_cp == cp_label).astype(np.uint8))

        return data

    data_cp = paste_data(paste_data(data_cp, 'left'), 'right')

    pasted_img = nib.Nifti1Image(data_cp, cp_img.affine, cp_img.header)
    nib.save(pasted_img, '{}/{}/{}_hs_sp_seg_pasted.nii'.format(subjects_folder, subject, subject))

def paste_pred(subject):
    cp_img = nib.load('{}/{}/{}_hs_cp_seg.nii'.format(subjects_folder, subject, subject))
    data_cp = cp_img.get_fdata(dtype=np.float32)

    sp_img = nib.load('{}/{}/{}_hs_sp_seg_pred_C40.nii'.format(subjects_folder, subject, subject))
    data_sp = sp_img.get_fdata(dtype=np.float32)
    data_sp = np.squeeze(data_sp)

    directions_dict = {'left': [4, 160, 42], 'right': [5, 161, 1]}
    parts_dict = {'iz': [160, 161], 'sp': [4, 5], 'cp': [42, 1]}

    # Plan #
    # CP: 161 -> 5
    # Paste SP 161 to CP

    def paste_data(data, region):
        sp_label, iz_label, cp_label = regions_dict[region]
        iz_labels = parts_dict['iz']

        # sp_pred = np.where((data == iz_labels[0]) | (data == iz_labels[1]) | (data == 0), sp_label, data)
        iz_pred_mask = ((data == iz_labels[0]) | (data == iz_labels[1]) | (data == 0)).astype(np.uint8)

        iz1_cond_mask = np.bitwise_and(iz_mask, (data_cp == iz_labels[0]).astype(np.uint8))
        iz2_cond_mask = np.bitwise_and(iz_mask, (data_cp == iz_labels[1]).astype(np.uint8))
        bak_cond_mask = np.bitwise_and(iz_mask, (data_cp == 0).astype(np.uint8))

        data = np.where(((iz1_cond_mask == 1) | (iz2_cond_mask == 1)) & (no_cond_mask != 1), iz_label, data)

        # iz_mask = (data == iz_label).astype(np.uint8)
        # er_mask = transform_mask(iz_mask, 'erode')
        # er_cp_mask = np.bitwise_and(er_mask, (data_cp == cp_label).astype(np.uint8))

        return data

    data_cp = paste_data(paste_data(data_cp, 'inner'), 'right')

    pasted_img = nib.Nifti1Image(data_cp, cp_img.affine, cp_img.header)
    nib.save(pasted_img, '{}/{}/{}_hs_sp_seg_pasted.nii'.format(subjects_folder, subject, subject))

subjects_folder = 'upsamp/batch3'
# Smooth borders
sigma = 1.3
tau = 1.3

for subject in ['FCB012', 'FCB100']: # os.listdir(subjects_folder):
    print()
    print(subject)

    curr_folder = '{}/{}/{}'.format(core_path, subjects_folder, subject)
    ls_seg_path = '{}/{}_ls_sp_seg.nii'.format(curr_folder, subject)
    hs_seg_path = '{}/{}_hs_sp_seg_upsamp.nii'.format(curr_folder, subject)
    hs_cp_seg_path = '{}/{}_hs_cp_seg.nii'.format(curr_folder, subject)

    ls_rec_path = '{}/{}_ls_nuc.nii'.format(curr_folder, subject)
    hs_rec_path = '{}/{}_hs_nuc.nii'.format(curr_folder, subject)

    # Alignment
    hs_rec_upsamp_path = '{}/{}_hs_nuc_upsamp.nii'.format(curr_folder, subject)
    hs_rec_verify_path = '{}/{}_hs_nuc_verify.nii'.format(curr_folder, subject)
    hs_rec_matrix_path = '{}/{}_hs_nuc_matrix.xfm'.format(curr_folder, subject)
    hs_seg_aligne_path = '{}/{}_hs_sp_seg_upsamp_al.nii'.format(curr_folder, subject)

    print('Up-sampling...')
    if 1: # os.path.exists(hs_seg_path):
        os.system('~/arch/Linux64/packages/irtk2/resample {} {} -size 0.5 0.5 0.5 -sbased >/dev/null 2>&1;'.format(ls_seg_path, hs_seg_path))
    if 1: # os.path.exists(hs_rec_upsamp_path):
        os.system('~/arch/Linux64/packages/irtk2/resample {} {} -size 0.5 0.5 0.5 -linear;'.format(ls_rec_path, hs_rec_upsamp_path))

    # xfm 
    print('Alignment...')
    if 1: # os.path.exists(hs_seg_aligne_path):
        os.system('~/arch/Linux64/packages/fsl/6.0/bin/flirt -in {} -ref {} -out {} -omat {} -searchrx -180 180 -searchry -180 180 -searchrz -180 180;'.
                  format(hs_rec_upsamp_path, hs_rec_path, hs_rec_verify_path, hs_rec_matrix_path))
        os.system('~/arch/Linux64/packages/fsl/6.0/bin/flirt -in {} -ref {} -out {} -applyxfm -interp nearestneighbour -init {};'.
                  format(hs_seg_path, hs_seg_path, hs_seg_aligne_path, hs_rec_matrix_path))
        
        if os.path.exists(hs_rec_verify_path):
            os.remove(hs_rec_verify_path)
        if os.path.exists(hs_seg_aligne_path):
            os.remove(hs_seg_aligne_path)
        os.system('gunzip {}.gz'.format(hs_seg_aligne_path)) # Gunzipping aligned Segmentation
        os.system('gunzip {}.gz'.format(hs_rec_verify_path)) # Gunzipping verified Reconstruction

    print('Smoothing...')
    smooth(subject, sigma=sigma, tau=tau)

    print('Pasting...')
    paste(subject)