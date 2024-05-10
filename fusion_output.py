import numpy as np
import nibabel as nib
import os

def fusion(subject, sp_name, cp_name):
    cp_img = nib.load(cp_name)
    data_cp = cp_img.get_fdata(dtype=np.float32)

    sp_img = nib.load(sp_name)
    data_sp = sp_img.get_fdata(dtype=np.float32)
    # data_sp = np.squeeze(data_sp)

    side_dict = {'left': [4, 160, 42], 'right': [5, 161, 1]}
    parts_dict = {'iz': [160, 161], 'sp': [4, 5], 'cp': [42, 1]}
    side_labels = {'left': 50, 'right': 100}
    opp_dict = {1: 42, 160: 161, 4:5, 42:1, 161: 160, 5:4}

    # 50 -> Left label
    # 100 -> Right label

    for label in [1, 42, 160, 161]:
        back_mask = np.bitwise_and((data_sp == 0), (data_cp == label))
        data_sp[back_mask] = label

    for side in ['left', 'right']:
        for label in side_dict[side]:
            data_cp = np.where((data_cp == label), side_labels[side], data_cp)
    
    cp_img = nib.Nifti1Image(data_cp, cp_img.affine, cp_img.header)
    nib.save(cp_img, '{}/side_mask.nii'.format(curr_folder))

    def correct_data(sp, cp, lab):

        current_side = 'left' if lab in side_dict['left'] else 'right'
        opposite_side = 'left' if current_side == 'right' else 'right'
        
        opposite_side_label = side_labels[opposite_side]
        opp_lab = opp_dict[lab]

        current_sp_mask = (sp == lab)
        opposite_side_mask = (cp == opposite_side_label)

        badside_mask = np.bitwise_and(current_sp_mask, opposite_side_mask)
        # sp = np.where(badside_mask, lab, sp)
        sp[badside_mask] = opp_lab

        return sp
    
    for label in [1, 42, 160, 161, 4, 5]:
        data_sp = correct_data(data_sp, data_cp, label)

    sp_img = nib.Nifti1Image(data_sp, sp_img.affine, sp_img.header)
    nib.save(sp_img, '{}/predC40_segmentation_fusion.nii'.format(curr_folder))


core_path = '/neuro/labs/grantlab/research/MRI_processing/milton.candela/highres_subplate'
subjects_folder = 'reviewed/pending'

for subject in os.listdir(subjects_folder):
    print()
    print(subject)

    if subject in ['4981517', '4489721']:
        continue

    curr_folder = '{}/{}/{}'.format(core_path, subjects_folder, subject)
    sp_model_pred = '{}/predC40_segmentation.nii'.format(curr_folder)
    cp_model_pred = '{}/predcp_segmentation.nii'.format(curr_folder)

    print('Pasting...')
    fusion(subject, sp_model_pred, cp_model_pred)