#!/usr/bin/env python3

import numpy as np
import glob,os,sys
import argparse, tempfile
from sp_utils import *
import tensorflow as tf
sys.path.append(os.path.dirname(__file__))

parser = argparse.ArgumentParser('   ==========   Fetal AU_Net segmentation script for high resolution input made by Sungmin You (11.27.22 ver.0)   ==========   ')
parser.add_argument('-input', '--input_MR',action='store',dest='inp',type=str, required=True, help='input MR file name (\'.nii or .nii.gz\') or folder name')
parser.add_argument('-output', '--output_loc',action='store',dest='out', type=str, required=True, help='Output path')
parser.add_argument('-name', '--model_name', action='store', dest='model_name', type=str, help='Model name')
parser.add_argument('-view', choices=['axi', 'cor', 'sag'], action='store', dest='view', required=True)
# parser.add_argument('-axi', '--axi_weight',action='store',dest='axi',default=os.path.dirname(os.path.abspath(__file__))+'/weights/axi.h5',type=str, help='Axial weight file')
# parser.add_argument('-cor', '--cor_weight',action='store',dest='cor',default=os.path.dirname(os.path.abspath(__file__))+'/weights/cor.h5',type=str, help='Coronal weight file')
# parser.add_argument('-sag', '--sag_weight',action='store',dest='sag',default=os.path.dirname(os.path.abspath(__file__))+'/weights/sag.h5',type=str, help='Sagittal weight file')
parser.add_argument('-gpu', default ='-1', type=str, help='GPU selection')
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

weights_path = '{}/weights/{}.h5'.format(args.model_name, args.view)
view = args.view

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

if os.path.isdir(args.inp):
    img_list = np.asarray(sorted(glob.glob(args.inp+'/*.nii*')))
elif os.path.isfile(args.inp):
    img_list = np.asarray(sorted(glob.glob(args.inp)))
else:
    img_list = np.asarray(sorted(glob.glob(args.inp)))


if len(img_list)==0:
    print('No such file or dictectory')
    exit()

mask= os.path.dirname(os.path.abspath(__file__))+'/down_mask-31_dil5.nii'

def gen_result(data_filter, input_name, result_loc):
    import nibabel as nib

    view_list = sorted(glob.glob(data_filter))
    view = nib.load(view_list[0])

    backgro = np.zeros(np.shape(view.get_fdata()))
    
    left_iz = np.zeros(np.shape(view.get_fdata()))
    right_iz = np.zeros(np.shape(view.get_fdata()))
    
    left_sp = np.zeros(np.shape(view.get_fdata()))
    right_sp = np.zeros(np.shape(view.get_fdata()))
    
    left_cp = np.zeros(np.shape(view.get_fdata()))    
    right_cp = np.zeros(np.shape(view.get_fdata()))

    _plate = np.zeros(np.shape(view.get_fdata()))
    total = np.zeros(np.shape(view.get_fdata()))

    for i in range(len(view_list)):
        view_data = nib.load(view_list[i]).get_fdata()

        loc = np.where(view_data==0)
        backgro[loc] = backgro[loc]+1

        loc = np.where(view_data==1)
        left_iz[loc] = left_iz[loc]+1
        loc = np.where(view_data==2)
        right_iz[loc] = right_iz[loc]+1

        loc = np.where(view_data==3)
        left_cp[loc] = left_cp[loc]+1
        loc = np.where(view_data==4)
        right_cp[loc] = right_cp[loc]+1

        loc = np.where(view_data==5)
        left_sp[loc] = left_sp[loc]+1
        loc = np.where(view_data==6)
        right_sp[loc] = right_sp[loc]+1

    result = np.concatenate((backgro[np.newaxis,:], left_iz[np.newaxis,:], right_iz[np.newaxis,:],
                             left_sp[np.newaxis,:], right_sp[np.newaxis,:], left_cp[np.newaxis,:], right_cp[np.newaxis,:]),axis=0)
    result = np.argmax(result, axis=0)

    # relabel
    ori_label = np.array([1, 2, 3,4, 5,6])
    relabel = np.array([161,160,1,42,5,4])
    for itr in range(len(ori_label)):
        loc = np.where((result>ori_label[itr]-0.5)&(result<ori_label[itr]+0.5))
        result[loc]=relabel[itr]
    filename=input_name.split('/')[-1:][0]
    filename=filename.split('.nii')[0]
    filename=filename+'_deep.nii.gz'
    new_img = nib.Nifti1Image(result, view.affine, view.header)
    nib.save(new_img, result_loc+'/'+filename)
    print('Prediction finishied!')
    print('save file : '+result_loc+'/'+filename)


with tempfile.TemporaryDirectory() as tmpdir:
    #tmpdir = args.out + '/debug'
    os.makedirs(tmpdir, exist_ok=True)

    if args.view == 'axi':
        test_dic, _ =make_dic(img_list, img_list, mask, 'axi', 0)
        model = Unet_network([192,192,1], 7, ite=3, depth=4).build()
        model.load_weights(weights_path)

        tmask = model.predict(test_dic)
        make_result(tmask,img_list,mask,tmpdir+'/','axi')
        tmask = model.predict(test_dic[:,::-1,:,:])
        make_result(tmask[:,::-1,:,:],img_list,mask,tmpdir+'/','axi','f1')
        tmask = model.predict(axfliper(test_dic))
        make_result(axfliper(tmask,1),img_list,mask,tmpdir+'/','axi','f2')
        tmask = model.predict(axfliper(test_dic[:,::-1,:,:]))
        make_result(axfliper(tmask[:,::-1,:,:],1),img_list,mask,tmpdir+'/','axi','f3')

        del model, tmask, test_dic
    elif args.view == 'cor':
        test_dic, _ =make_dic(img_list, img_list, mask, 'cor', 0)
        model = Unet_network([192,192,1], 7,ite=3, depth=4).build()
        model.load_weights(weights_path)

        tmask = model.predict(test_dic)
        make_result(tmask,img_list,mask,tmpdir+'/','cor')
        tmask = model.predict(test_dic[:,:,::-1,:])
        make_result(tmask[:,:,::-1,:],img_list,mask,tmpdir+'/','cor','f1')
        tmask = model.predict(cofliper(test_dic))
        make_result(cofliper(tmask,1),img_list,mask,tmpdir+'/','cor','f2')
        tmask = model.predict(cofliper(test_dic[:,:,::-1,:]))
        make_result(cofliper(tmask[:,:,::-1,:],1),img_list,mask,tmpdir+'/','cor','f3')

        del model, tmask, test_dic
    elif args.view == 'sag':
        test_dic, _ =make_dic(img_list, img_list, mask, 'sag', 0)
        model = Unet_network([192,192,1], 4, ite=3, depth=4).build()
        model.load_weights(weights_path)

        tmask = model.predict(test_dic)
        make_result(tmask,img_list,mask,tmpdir+'/','sag')
        tmask = model.predict(test_dic[:,::-1,:,:])
        make_result(tmask[:,::-1,:,:],img_list,mask,tmpdir+'/','sag','f1')
        tmask = model.predict(test_dic[:,:,::-1,:])
        make_result(tmask[:,:,::-1,:],img_list,mask,tmpdir+'/','sag','f2')

        del model, tmask, test_dic

    if np.shape(img_list):
        for i2 in range(len(img_list)): 
            filename=img_list[i2].split('/')[-1:][0]
            filename=filename.split('.nii')[0]
            gen_result(tmpdir+'/'+filename+'*'+args.view+'*', img_list[i2], args.out+'/')
            # make_sum(tmpdir+'/'+filename+'*axi*', tmpdir+'/'+filename+'*cor*',tmpdir+'/'+filename+'*sag*', img_list[i2], args.out+'/')
            make_verify(img_list[i2], args.out+'/')
    else:
        # make_sum(tmpdir+'/'+filename+'*axi*', tmpdir+'/'+filename+'*cor*',tmpdir+'/'+filename+'*sag*', img_list, args.out+'/')
        make_verify(img_list, args.out+'/')