import os

batches = ['batch1', 'batch2', 'batch3', 'batch4']

destiny_folder = 'reviewed'
initial_folder = 'upsamp'

def create_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)
    else:
        print('{} already existed!'.format(dir))

create_dir(destiny_folder)
create_dir(destiny_folder+'/done')
create_dir(destiny_folder+'/pending')

for batch in ['batch4']: # batches:
    subjects = os.listdir('{}/{}'.format(initial_folder, batch))

    for subject in subjects:
        print(subject)

        if subject != '2123603':
            pass

        status = 'pending' if batch == 'batch4' else 'done'
        create_dir('{}/{}/{}'.format(destiny_folder, status, subject))

        if status == 'done':
            os.system('cp {}/{}/{}/{}_hs_nuc.nii {}/{}/{}/recon_to31_nuc.nii ;'.format(initial_folder, batch, subject, subject, destiny_folder, status, subject))
            os.system('cp {}/{}/{}/{}_hs_sp_seg.nii {}/{}/{}/segmentation_to31_final.nii ;'.format(initial_folder, batch, subject, subject, destiny_folder, status, subject))
        elif status == 'pending':
            os.system('cp {}/{}/{}/{}_hs_nuc.nii {}/{}/{}/recon_to31_nuc.nii ;'.format(initial_folder, batch, subject, subject, destiny_folder, status, subject))

            os.system('cp {}/{}/{}/{}_ls_nuc.nii {}/{}/{}/lowres_recon_to31_nuc.nii ;'.format(initial_folder, batch, subject, subject, destiny_folder, status, subject))
            os.system('cp {}/{}/{}/{}_ls_sp_seg.nii {}/{}/{}/lowres_segmentation.nii ;'.format(initial_folder, batch, subject, subject, destiny_folder, status, subject))

            os.system('cp {}/{}/{}/{}_hs_cp_seg.nii {}/{}/{}/predcp_segmentation.nii ;'.format(initial_folder, batch, subject, subject, destiny_folder, status, subject))
            os.system('cp {}/{}/{}/{}_hs_sp_seg_pasted.nii {}/{}/{}/pasted_segmentation.nii ;'.format(initial_folder, batch, subject, subject, destiny_folder, status, subject))
            os.system('cp {}/{}/{}/{}_hs_sp_seg_pred_C40.nii {}/{}/{}/predC40_segmentation.nii ;'.format(initial_folder, batch, subject, subject, destiny_folder, status, subject))
            os.system('cp {}/{}/{}/{}_hs_sp_seg_pred_A40.nii {}/{}/{}/predA40_segmentation.nii ;'.format(initial_folder, batch, subject, subject, destiny_folder, status, subject))
            os.system('cp {}/{}/{}/{}_hs_sp_seg_pred_C40F.nii {}/{}/{}/predC40F_segmentation.nii ;'.format(initial_folder, batch, subject, subject, destiny_folder, status, subject))
    

