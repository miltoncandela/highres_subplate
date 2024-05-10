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
parser.add_argument('-weights', '--weights_loc', action = 'store', dest='wei', type =str, required=True, help='Weights path')
# parser.add_argument('-axi', '--axi_weight',action='store',dest='axi',default=os.path.dirname(os.path.abspath(__file__))+'/weights/axi.h5',type=str, help='Axial weight file')
# parser.add_argument('-cor', '--cor_weight',action='store',dest='cor',default=os.path.dirname(os.path.abspath(__file__))+'/weights/cor.h5',type=str, help='Coronal weight file')
# parser.add_argument('-sag', '--sag_weight',action='store',dest='sag',default=os.path.dirname(os.path.abspath(__file__))+'/weights/sag.h5',type=str, help='Sagittal weight file')
parser.add_argument('-gpu', default ='-1', type=str, help='GPU selection')
args = parser.parse_args()



# python3 predict_dataTTA.py --input_MR upsamp/done/FCB061/FCB061_hs_nuc.nii --output_loc C -axi C/C40_auto/weights/axi.h5 -cor C/C40_auto/weights/cor.h5 -sag -axi C/C40_auto/weights/sag.h5 -gpu 1
# python3 predict_dataTTA.py --input_MR {}/FCB061_hs_nuc.nii --output_loc {} --weights_loc C/C40_auto -gpu 0


axi_weights = args.wei+'/weights/axi.h5'
cor_weights = args.wei+'/weights/cor.h5'
sag_weights = args.wei+'/weights/sag.h5'

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

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

with tempfile.TemporaryDirectory() as tmpdir:
    #tmpdir = args.out + '/debug'
    os.makedirs(tmpdir, exist_ok=True)
    test_dic, _ =make_dic(img_list, img_list, mask, 'axi', 0)
    model = Unet_network([192,192,1], 7,ite=3, depth=4).build()
    model.load_weights(axi_weights)

    #model.summary()

    tmask = model.predict(test_dic)
    make_result(tmask,img_list,mask,tmpdir+'/','axi')
    tmask = model.predict(test_dic[:,::-1,:,:])
    make_result(tmask[:,::-1,:,:],img_list,mask,tmpdir+'/','axi','f1')
    tmask = model.predict(axfliper(test_dic))
    make_result(axfliper(tmask,1),img_list,mask,tmpdir+'/','axi','f2')
    tmask = model.predict(axfliper(test_dic[:,::-1,:,:]))
    make_result(axfliper(tmask[:,::-1,:,:],1),img_list,mask,tmpdir+'/','axi','f3')

    del model, tmask, test_dic

    test_dic, _ =make_dic(img_list, img_list, mask, 'cor', 0)
    model = Unet_network([192,192,1], 7,ite=3, depth=4).build()
    model.load_weights(cor_weights)

    tmask = model.predict(test_dic)
    make_result(tmask,img_list,mask,tmpdir+'/','cor')
    tmask = model.predict(test_dic[:,:,::-1,:])
    make_result(tmask[:,:,::-1,:],img_list,mask,tmpdir+'/','cor','f1')
    tmask = model.predict(cofliper(test_dic))
    make_result(cofliper(tmask,1),img_list,mask,tmpdir+'/','cor','f2')
    tmask = model.predict(cofliper(test_dic[:,:,::-1,:]))
    make_result(cofliper(tmask[:,:,::-1,:],1),img_list,mask,tmpdir+'/','cor','f3')

    del model, tmask, test_dic


    test_dic, _ =make_dic(img_list, img_list, mask, 'sag', 0)
    model = Unet_network([192,192,1], 4, ite=3, depth=4).build()
    model.load_weights(sag_weights)

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
            make_sum(tmpdir+'/'+filename+'*axi*', tmpdir+'/'+filename+'*cor*',tmpdir+'/'+filename+'*sag*', img_list[i2], args.out+'/')
            make_verify(img_list[i2], args.out+'/')
    else:
        #make_sum(tmpdir+'/'+filename+'*axi*', tmpdir+'/'+filename+'*cor*',tmpdir+'/'+filename+'*sag*', img_list, args.out+'/')
        make_verify(img_list, args.out+'/')