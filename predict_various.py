import os

weights_name = {'C/C40_auto': 'C40', 'C/C40_auto_fine': 'C40F', 'A/A40_auto': 'A40'}
source_folder = 'upsamp/done'
for subject in os.listdir(source_folder):
    for weights in ['C/C40_auto', 'C/C40_auto_fine', 'A/A40_auto']:
        print(subject, weights)
        subject_folder = source_folder+'/{}'.format(subject)
        os.system('python3 predict_dataTTA.py --input_MR {}/{}_hs_nuc.nii --output_loc {} --weights_loc {} -gpu 0'.format(subject_folder, subject,
                                                                                                                          subject_folder, weights))
        os.system('gunzip {}/{}_hs_nuc_deep_agg.nii.gz'.format(subject_folder, subject))
        os.system('mv {}/{}_hs_nuc_deep_agg.nii {}/{}_hs_sp_seg_pred_{}.nii'.format(subject_folder, subject, subject_folder, subject, weights_name[weights]))
    print()
