import pandas as pd
import os

sp_path = '/neuro/labs/grantlab/research/MRI_processing/milton.candela/fetal_subplate'

batch_processed = 3
if batch_processed == 3:
    ''' Batch 3: CHD '''
    destiny_path = '/neuro/labs/grantlab/research/MRI_processing/milton.candela/highres_subplate/upsamp/batch3'

    df = pd.read_csv('/neuro/labs/grantlab/research/MRI_processing/milton.candela/highres_subplate/upsamp/batch3_info.csv')

    FCB_to_bch = dict(zip(df.dropna(axis=0)['ID'], df.dropna(axis=0)['Subject']))
    bch_to_MRN = {'BCH_0040_s1': 2123603, 'BCH_0041_s1': 4489721, 'BCH_0042_s1': 4981517}

    for row in range(df.shape[0]):
        curr_id = df.iloc[row, 1]
        print('Processing {} ...'.format(curr_id))

        if (curr_id not in FCB_to_bch.keys()) and (not curr_id.isdigit()): # NaN FCBXXX
            subject_path = '{}/{}'.format(destiny_path, curr_id)
            if not os.path.exists(subject_path):
                os.mkdir(subject_path)

            junshen_path = df.iloc[row, 3]

            if os.path.exists('{}/{}_hs_cp_seg.nii.gz'.format(subject_path, curr_id)):
                os.system('rm {}/{}_hs_cp_seg.nii.gz'.format(subject_path, curr_id))
            
            os.system('cp {}/recon_to31_nuc_deep_agg.nii.gz {}/{}_hs_cp_seg.nii.gz ;'.format(junshen_path, subject_path, curr_id))

            if os.path.exists('{}/{}_hs_cp_seg.nii'.format(subject_path, curr_id)):
                os.system('rm {}/{}_hs_cp_seg.nii'.format(subject_path, curr_id))

            os.system('gunzip {}/{}_hs_cp_seg.nii.gz'.format(subject_path, curr_id))
            os.system('cp {}/recon_to31_nuc.nii {}/{}_hs_nuc.nii ;'.format(junshen_path, subject_path, curr_id))

            os.system('cp {}/input/{}_nuc_deep_subplate_dilate.nii {}/{}_ls_sp_seg.nii ;'.format(sp_path, curr_id, subject_path, curr_id))
            os.system('cp {}/recons/{}_nuc.nii {}/{}_ls_nuc.nii ;'.format(sp_path, curr_id, subject_path, curr_id))

        else: # BCH_XXXX_s1 FCBXXX
            if not curr_id.isdigit():
                FCB_id = curr_id
                BCH_id = FCB_to_bch[FCB_id]
                MRN_id = bch_to_MRN[BCH_id]
            else:
                MRN_id = curr_id
                BCH_id = df.iloc[row, 0]

            subject_path = '{}/{}'.format(destiny_path, MRN_id)        
            if not os.path.exists(subject_path):
                os.mkdir(subject_path)

            junshen_path = df.iloc[row, 3]

            if os.path.exists('{}/{}_hs_cp_seg.nii.gz'.format(subject_path, MRN_id)):
                os.system('rm {}/{}_hs_cp_seg.nii.gz'.format(subject_path, MRN_id))
            
            os.system('cp {}/recon_to31_nuc_deep_agg.nii.gz {}/{}_hs_cp_seg.nii.gz ;'.format(junshen_path, subject_path, MRN_id))

            if os.path.exists('{}/{}_hs_cp_seg.nii'.format(subject_path, MRN_id)):
                os.system('rm {}/{}_hs_cp_seg.nii'.format(subject_path, MRN_id))

            os.system('gunzip {}/{}_hs_cp_seg.nii.gz'.format(subject_path, MRN_id))
            os.system('cp {}/recon_to31_nuc.nii {}/{}_hs_nuc.nii ;'.format(junshen_path, subject_path, MRN_id))

            os.system('cp {}/input/{}_nuc_deep_subplate_dilate.nii {}/{}_ls_sp_seg.nii ;'.format(sp_path, BCH_id, subject_path, MRN_id))
            os.system('cp {}/recons/{}_nuc.nii {}/{}_ls_nuc.nii ;'.format(sp_path, BCH_id, subject_path, MRN_id))


elif batch_processed == 2:
    ''' Batch 2: CHD '''
    destiny_path = '/neuro/labs/grantlab/research/MRI_processing/milton.candela/highres_subplate/upsamp/batch2'

    df = pd.read_csv('/neuro/labs/grantlab/research/MRI_processing/milton.candela/highres_subplate/upsamp/batch2_info.csv')
    print()

    FCB_to_bch = dict(zip(df.dropna(axis=0)['ID'], df.dropna(axis=0)['Subject']))
    bch_to_MRN = {'BCH_0031_s1': 4841657, 'BCH_0039_s1': 2005014, 'BCH_0043_s1': 5045113}


    for row in range(df.shape[0]):
        curr_id = df.iloc[row, 1]
        print('Processing {} ...'.format(curr_id))

        if curr_id not in FCB_to_bch.keys():
            subject_path = '{}/{}'.format(destiny_path, curr_id)
            if not os.path.exists(subject_path):
                os.mkdir(subject_path)

            junshen_path = df.iloc[row, 3]

            if os.path.exists('{}/{}_hs_cp_seg.nii.gz'.format(subject_path, curr_id)):
                os.system('rm {}/{}_hs_cp_seg.nii.gz'.format(subject_path, curr_id))
            
            os.system('cp {}/recon_to31_nuc_deep_agg.nii.gz {}/{}_hs_cp_seg.nii.gz ;'.format(junshen_path, subject_path, curr_id))

            if os.path.exists('{}/{}_hs_cp_seg.nii'.format(subject_path, curr_id)):
                os.system('rm {}/{}_hs_cp_seg.nii'.format(subject_path, curr_id))

            os.system('gunzip {}/{}_hs_cp_seg.nii.gz'.format(subject_path, curr_id))
            os.system('cp {}/recon_to31_nuc.nii {}/{}_hs_nuc.nii ;'.format(junshen_path, subject_path, curr_id))

            os.system('cp {}/input/{}_nuc_deep_subplate_dilate.nii {}/{}_ls_sp_seg.nii ;'.format(sp_path, curr_id, subject_path, curr_id))
            os.system('cp {}/recons/{}_nuc.nii {}/{}_ls_nuc.nii ;'.format(sp_path, curr_id, subject_path, curr_id))

        else:
            FCB_id = curr_id
            BCH_id = FCB_to_bch[FCB_id]
            MRN_id = bch_to_MRN[BCH_id]

            subject_path = '{}/{}'.format(destiny_path, MRN_id)        
            if not os.path.exists(subject_path):
                os.mkdir(subject_path)

            junshen_path = df.iloc[row, 3]

            if os.path.exists('{}/{}_hs_cp_seg.nii.gz'.format(subject_path, MRN_id)):
                os.system('rm {}/{}_hs_cp_seg.nii.gz'.format(subject_path, MRN_id))
            
            os.system('cp {}/recon_to31_nuc_deep_agg.nii.gz {}/{}_hs_cp_seg.nii.gz ;'.format(junshen_path, subject_path, MRN_id))

            if os.path.exists('{}/{}_hs_cp_seg.nii'.format(subject_path, MRN_id)):
                os.system('rm {}/{}_hs_cp_seg.nii'.format(subject_path, MRN_id))

            os.system('gunzip {}/{}_hs_cp_seg.nii.gz'.format(subject_path, MRN_id))
            os.system('cp {}/recon_to31_nuc.nii {}/{}_hs_nuc.nii ;'.format(junshen_path, subject_path, MRN_id))

            os.system('cp {}/input/{}_nuc_deep_subplate_dilate.nii {}/{}_ls_sp_seg.nii ;'.format(sp_path, BCH_id, subject_path, MRN_id))
            os.system('cp {}/recons/{}_nuc.nii {}/{}_ls_nuc.nii ;'.format(sp_path, BCH_id, subject_path, MRN_id))


def copy_files(dest_dir):
    print()
    print('Dest dir:', dest_dir)
    print()

    cp_path = '/neuro/labs/grantlab/research/MRI_processing/sungmin.you/SM_codes/fetal_CP_segmentation/high_res_exp/training_test'
    sp_path = '/neuro/labs/grantlab/research/MRI_processing/milton.candela/fetal_subplate'

    files_cp = os.listdir(cp_path + '/Upsample_MR')
    cp_file_dict = dict(zip([x.split('_')[3] for x in files_cp], [x[15:] for x in files_cp]))

    def pross_subject(curr_folder, cp_file, sp_file, dest_file):

        # HyukJin
        # cp_name = 'segmentation_to31_final_highres_HJ.nii' if os.path.exists('{}/{}/recon_segmentation/segmentation_to31_final_highres_HJ.nii'.format(cp_path, cp_file)) else 'segmentation_to31_final_HJ.nii'
        # os.system('cp {}/{}/recon_segmentation/{} {}/{}_hs_cp_seg.nii ;'.format(cp_path, cp_file, cp_name, curr_folder, dest_file))
        # os.system('cp {}/{}/recon_segmentation/recon_to31_nuc.nii {}/{}_hs_nuc.nii ;'.format(cp_path, cp_file, curr_folder, dest_file))

        cp_name = cp_file_dict[cp_file]

        # Sungmin
        os.system('cp {}/Upsample_GT_LAS/segmentation_to31_final_highres_HJ_LAS_{} {}/{}_hs_cp_seg.nii ;'.format(cp_path, cp_name, curr_folder, dest_file)) # Segmentation
        os.system('cp {}/Upsample_MR/recon_to31_nuc_{} {}/{}_hs_nuc.nii ;'.format(cp_path, cp_name, curr_folder, dest_file)) # Recon

        os.system('cp {}/input/{}_nuc_deep_subplate_dilate.nii {}/{}_ls_sp_seg.nii ;'.format(sp_path, sp_file, curr_folder, dest_file))
        os.system('cp {}/recons/{}_nuc.nii {}/{}_ls_nuc.nii ;'.format(sp_path, sp_file, curr_folder, dest_file))

    if os.path.exists(dest_dir):
        # os.mkdir(dest_dir)
        for subject in list(sp_MRN_BCH.keys()) + ov_FCB_NUM:
            if str(subject)[:3] == 'FCB':
                # FCB
                file_long = subject
                file_short = subject.split('/')[0]
                file_destiny = file_short
            else:
                # MRN
                file_long = cp_MRN_BCH[subject]
                file_short = sp_MRN_BCH[subject]
                file_destiny = subject

            print('Processing {} ...'.format(file_short))

            curr_folder = '{}/{}'.format(dest_dir, file_destiny)

            if not os.path.exists(curr_folder):
                print('Copying', curr_folder.split('/')[1], '...')
                os.mkdir(curr_folder)
                pross_subject(curr_folder, file_long, file_short, file_destiny)
            else:
                print(curr_folder.split('/')[1], 'already copied!')
            print()
    else:
        print('Dest_dir already exists!')

# copy_files('upsamp')
