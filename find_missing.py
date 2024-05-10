import os

destiny_folder = '/neuro/labs/grantlab/research/MRI_processing/milton.candela/highres_subplate/missing_data'
recon_path = '/neuro/labs/grantlab/research/MRI_processing/milton.candela/fetal_subplate/recons'

placenta_dict = {'359034': 359034, '5285137' : 5285137}
chd_dict = {'BCH_0026_s1': 'FCB010', # 4738892
            'BCH_0029_s1': 'FCB012', # 4757442
            'BCH_0040_s1': 'FCB100', # 2123603
            'BCH_0042_s1': 'FCB092', # 4981517
            'FCB054': 'FCB054',
            'FCB067': 'FCB067',
            'FCB129': 'FCB129'}
failed_chd_dict = {'BCH_0041_s1': 'FCB083'} # 4489721
normative_dict = {'BCH_0044_s1' : 4991273,
                  'BCH_0045_s1' : 5059637,
                  'BCH_0046_s1' : 5097923,
                  'BCH_0050_s1' : 5140282,
                  'BCH_0051_s1' : 5149271,
                  'BCH_0052_s1' : 5204150,
                  'BCH_0055_s1' : 5066879,
                  'BCH_0056_s1' : 5089529,
                  'BCH_0057_s1' : 2035695, 
                  'BCH_0058_s1' : 5153010,
                  'BCH_0059_s1' : 5069056,
                  'BCH_0060_s1' : 5076822, 
                  'BCH_0061_s1' : 5091951,
                  'BCH_0062_s1' : 5117677,
                  'BCH_0063_s1' : 5138905,
                  'BCH_0064_s1' : 4620341,
                  'BCH_0065_s1' : 4821054,
                  'BCH_0067_s1' : 3003884}

def copy_files(source_path, subject_dict, protocol_name):
    print('* * * {} * * *'.format(protocol_name))
    if not os.path.exists('{}/{}'.format(destiny_folder, protocol_name)):
        os.mkdir('{}/{}'.format(destiny_folder, protocol_name))

    for subject in subject_dict.keys():
        protocol_id = subject_dict[subject]

        if subject.isdigit() and len(subject) < 7:
            protocol_id = '0' + str(protocol_id)

        if not os.path.exists('{}/{}/{}'.format(destiny_folder, protocol_name, protocol_id)):
            print('Processing {}...'.format(subject))
                
            os.system('cp -r {}/{} {}/{};'.format(source_path, protocol_id, destiny_folder, protocol_name))
        else:
            print('{} already processed!'.format(subject))

        sub_folder = os.listdir('{}/{}/{}'.format(destiny_folder, protocol_name, protocol_id))[0]

        if not os.path.exists('{}/{}/{}/{}/recon_to31_nuc.nii'.format(destiny_folder, protocol_name, protocol_id, sub_folder)):
            print('{}/{}_nuc.nii {}/{}/{}/{}/recon_to31_nuc.nii;'.format(recon_path, subject, destiny_folder, protocol_name, protocol_id, sub_folder))
            os.system('cp {}/{}_nuc.nii {}/{}/{}/{}/recon_to31_nuc.nii;'.format(recon_path, subject, destiny_folder, protocol_name, protocol_id, sub_folder))
            
    
    print()

copy_files('/neuro/labs/grantlab/research/MRI_processing/Data/Placenta_protocol', placenta_dict, 'Placenta')
copy_files('/neuro/labs/grantlab/research/MRI_processing/Data/Normative/Processed', normative_dict, 'Normative')
copy_files('/neuro/labs/grantlab/research/MRI_processing/Data/CHD_protocol/Processed', chd_dict, 'CHD')
copy_files('/neuro/labs/grantlab/research/MRI_processing/Data/CHD_protocol/Failed', failed_chd_dict, 'CHD')

failed_chd_dict

print(len(normative_dict.keys()))
exit()

pd.set_option('display.max_rows', None)

df = pd.read_csv('/neuro/users/mri.team/fetal_mri/Data/CHD_protocol/study_ID', sep='\t', header=None)
df.columns = ['FCB', 'MRN']
fcb_to_MRN = dict(zip(list(df.FCB), list(df.MRN)))

def pross_sub(s):
    if s[:3] == 'FCB':
        s = s.replace('-', '')
        if s in list(fcb_to_MRN.keys()):
            return fcb_to_MRN[s]
        else:
            return None
    elif (len(s.split('_')) > 1) or (s[:3] == 'sub') or (s[:2] == 'BM') or (s[-4:] == '.nii') or (len(s) == 3):
        return None
    else:
        return s

'''    
print("--- Sungmin's data ---")
core_path = '/neuro/labs/grantlab/research/MRI_processing/'

highres_nucs = core_path + 'sungmin.you/SM_codes/fetal_CP_segmentation/high_res_exp/training_test/Upsample_MR' # recon_to31_nuc_FCB001_{MRN}.nii
highres_segs = core_path + 'sungmin.you/SM_codes/fetal_CP_segmentation/high_res_exp/training_test/Upsample_GT_LAS' # segmentation_to31_final_highres_HJ_LAS_FCB001_{MRN}.nii

sub_hn = [file.split('_')[3] for file in os.listdir(highres_nucs)]
sub_hs = [file.split('_')[6] for file in os.listdir(highres_segs)]
'''

main_path = '/neuro/labs/grantlab/research/MRI_processing/'
placenta_path = 'Data/Placenta_protocol'
normative_path = 'Data/Normative/Processed'

placenta_hn = [int(file) for file in os.listdir(main_path + placenta_path) if file.isdigit()]
normativ_hn = [int(file) for file in os.listdir(main_path + normative_path) if file.isdigit()]

print('         Nuc')
print('Placenta', len(placenta_hn))
print('Normativ', len(normativ_hn))

df_placenta = pd.DataFrame(placenta_hn, index=range(len(placenta_hn)), columns=['MRN'])
df_placenta['Protocol'] = 'Placenta'

df_normativ = pd.DataFrame(normativ_hn, index=range(len(normativ_hn)), columns=['MRN'])
df_normativ['Protocol'] = 'Normative'

df_cp = pd.concat([df_placenta, df_normativ], axis=0)
hs_MRN = df_cp.MRN

print(df_cp.shape)

print('High-res CP MRN')
print(sorted(hs_MRN))
print()

df_sp = pd.read_csv('/neuro/labs/grantlab/research/MRI_processing/milton.candela/highres_subplate/sp_data_information.csv')
df_sp = df_sp.loc[~df_sp['Segmented by'].isna(), :].iloc[:, :3].loc[df_sp.MRN != '-', :]

def pross_MRN(l):
    # BCH, Anon, MRN
    try: return float(l[2])
    except ValueError: return fcb_to_MRN[l[1]] if l[1][:3] == 'FCB' else None
    
ls_MRN = [pross_MRN(list(df_sp.iloc[i, :])) for i in range(df_sp.shape[0])]
df_sp['MRN'] = ls_MRN

print('Low-res SP MRN')
print(sorted([x for x in ls_MRN if x is not None])) # Do not count BM (as they are marked as None, nan = FCB)
print(len([x for x in ls_MRN if x is not None]))
print(len(ls_MRN))
print()

# MRN_to_fcb = dict(zip(list(df.MRN), list(df.FCB)))
# print(dict(zip([x for x in ls_MRN if x in MRN_to_fcb.keys()], [MRN_to_fcb[x] for x in ls_MRN if  x in MRN_to_fcb.keys()])))

ov_MRN = list(set(ls_MRN).intersection(set(hs_MRN)))
print('Overlapped MRN:', len(ov_MRN), sorted(ov_MRN))

print(df_cp.loc[df_cp.MRN.isin(ov_MRN), :].sort_values('MRN'))

exit()

ov_FCB = list(set([x.split('/')[0] for x in df_cp.ID]).intersection(set(df_sp.Anon_number)))
print('Overlapped FCB:', len(ov_FCB), ov_FCB)

FCB_to_NUM = dict(zip(df_cp.ID.map(lambda x : x.split('/')[0]), df_cp.ID))
ov_FCB_NUM = [FCB_to_NUM[x].rstrip('//') for x in ov_FCB]
# FCB_GA = df_cp.loc[df_cp.ID.map(lambda x : x.rstrip('//')).isin(ov_FCB_NUM), ' GA']

df_cp['MRN'] = [float(x) for x in df_cp.MRN]

df_sp = df_sp.loc[df_sp.MRN.isin(ov_MRN),]
df_cp = df_cp.loc[df_cp.MRN.isin(ov_MRN),]

print()
print(df_cp.shape, df_sp.shape)

'''
plt.hist(list(df_cp[' GA']) + list(FCB_GA), bins = 7)
plt.xlabel('GA')
plt.ylabel('Frequency')
plt.title('High-res SP data GA distribution')
plt.xlim([16, 36])
plt.savefig('hist_overlap_GA.png')
'''

pross_BCH = lambda l: l[0] if type(l[0]) != 'str' else l[1]

df_sp['BCH_number'] = [pross_BCH(list(df_sp.iloc[i, :])) for i in range(df_sp.shape[0])]
df_cp['ID'] = [x.rstrip('//') for x in df_cp.ID]

# sp_FCB_BCH = dict(zip(df_sp.MRN, df_sp.BCH_number))
# cp_FCB_BCH = dict(zip([x.split('/')[0] for x in df_cp.MRN], df_cp.ID))

df_sp['MRN'] = [int(x) for x in df_sp.MRN]
df_cp['MRN'] = [int(x) for x in df_cp.MRN]

sp_MRN_BCH = dict(zip(df_sp.MRN, df_sp.BCH_number))
cp_MRN_BCH = dict(zip(df_cp.MRN, df_cp.ID))

print()
print(sp_MRN_BCH)
print(cp_MRN_BCH)

print()

print(sp_MRN_BCH)
print(ov_FCB_NUM)
print()

# sp_MRN_BCH = {key: sp_MRN_BCH[key] for key in sp_MRN_BCH.keys() if key in [4752588]}
# ov_FCB_NUM = ['FCB011']

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

copy_files('upsamp')

exit()

print("--- Sungmin's data ---")
core_path = '/neuro/labs/grantlab/research/MRI_processing/'

highres_nucs = core_path + 'sungmin.you/SM_codes/fetal_CP_segmentation/high_res_exp/training_test/Upsample_MR' # recon_to31_nuc_FCB001_{MRN}.nii
highres_segs = core_path + 'sungmin.you/SM_codes/fetal_CP_segmentation/high_res_exp/training_test/Upsample_GT_LAS' # segmentation_to31_final_highres_HJ_FCB001_{MRN}.nii
lowres_nucs = core_path + 'milton.candela/fetal_subplate/input' # FCB001_nuc.nii
lowres_segs = core_path + 'milton.candela/fetal_subplate/recons' # FCB001_nuc_deep_subplate_dilate.nii

sub_HN = [file.split('_')[3] for file in os.listdir(highres_nucs)]
sub_HS = [file.split('_')[5] for file in os.listdir(highres_segs)]

sub_LN = [file.split('_')[0] + file.split('_')[1] if file.split('_')[0] == 'BCH' else file.split('_')[0] for file in os.listdir(lowres_nucs)]
sub_LS = [file.split('_')[0] + file.split('_')[1] if file.split('_')[0] == 'BCH' else file.split('_')[0] for file in os.listdir(lowres_segs)]

print('  Nuc Seg')
print('HR', len(sub_HN), len(sub_HS))
print('LR', len(sub_LN), len(sub_LS))
print('')
print('Overlapped nuc:', set(sub_HN).intersection(set(sub_LN)))
print('Overlapped seg:', set(sub_HS).intersection(set(sub_LS)))

