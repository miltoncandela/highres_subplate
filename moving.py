import os

excluded_subjects = ['FCB011', '4845328', '4981517', 'FCB015']
subjects_dir = 'upsamp/done'
destiny_dir = 'data/set40/set40_nii'

if not os.path.exists(destiny_dir):
    os.mkdir(destiny_dir)
    os.mkdir(destiny_dir + '/Upsample_MR')
    os.mkdir(destiny_dir + '/Upsample_GT')

for subject in os.listdir(subjects_dir):
    # if subject not in excluded_subjects:
    os.system('cp {}/{}/{}_hs_nuc.nii {}/Upsample_MR/recon_to31_nuc_{}.nii ;'.format(subjects_dir, subject, subject, destiny_dir, subject))
    os.system('cp {}/{}/{}_hs_sp_seg.nii {}/Upsample_GT/segmentation_to31_final_highres_MC_{}.nii ;'.format(subjects_dir, subject, subject, destiny_dir, subject))
