#!/usr/bin/env python
import sys

import glob
import argparse
import numpy as np
import h5py
import pygame
import random
from PIL import Image
import matplotlib.pyplot as plt

import math
import time
import scipy
import os
from sklearn import preprocessing
from collections import deque
from skimage.transform import resize


def string_to_node(string):
  vec = string.split(',')

  return (int(vec[0]),int(vec[1]))

def string_to_floats(string):
  vec = string.split(',')

  return (float(vec[0]),float(vec[1]),float(vec[2]))

def read_all_files(file_names):
 
  dataset_names = ['targets']
  datasets_cat = [list([]) for _ in xrange(len(dataset_names))]

  lastidx = 0
  count =0
  #print file_names
  for cword in file_names:
    try:
        print cword
        print count
        dset = h5py.File(cword, "r")  

        for i in range(len(dataset_names)):

          dset_to_append = dset[dataset_names[i]]

          if dset_to_append.shape[1] >23 and dset_to_append.shape[1] <28:  # carla case
            zero_vec = np.zeros((dset_to_append.shape[0],1))

            dset_to_append = np.append(dset_to_append,zero_vec,axis=1)


          datasets_cat[i].append( dset_to_append[:])

        
        dset.flush()
        count +=1

    except IOError:
      import traceback
      exc_type, exc_value, exc_traceback  = sys.exc_info()
      traceback.print_exc()
      traceback.print_tb(exc_traceback,limit=1, file=sys.stdout)
      traceback.print_exception(exc_type, exc_value, exc_traceback,
                            limit=2, file=sys.stdout)
      print "failed to open", cword

  for i in range(len(dataset_names)):     
    datasets_cat[i] = np.concatenate(datasets_cat[i], axis=0)
    datasets_cat[i] = datasets_cat[i].transpose((1,0))

  return datasets_cat


# ***** main loop *****
if __name__ == "__main__":
  
  # Concatenate all files
  name = '2018127_mm45_rc28_wpz_M_mm41_cityscapes_aug_cluster_6' #'CVPR25Noise'
  dataset_dir = './dataset/'
  in_dir = '/home/matthias/Downloads/2018127_mm45_rc28_wpz_M_mm41_cityscapes_aug_cluster_6'

  out_dir = dataset_dir + name + '/'

  testPath = out_dir + 'test/'
  testAnnotPath = out_dir + 'testannot/'
  files = [os.path.join(in_dir, f) for f in glob.glob1(in_dir, "data_*.h5")]

  h5_start = 0
  h5_last = 50
  bRGB = True


  if not os.path.exists(out_dir):
    os.mkdir(out_dir)

  if not os.path.exists(testPath):
    os.mkdir(testPath)
  if not os.path.exists(testAnnotPath):
    os.mkdir(testAnnotPath)

  # Now go over all files   
  sequence_num = range(h5_start,h5_last+1)
  for h_num in sequence_num:

    print " SEQUENCE NUMBER ",h_num
    data = h5py.File(os.path.join(in_dir,'data_' + str(h_num).zfill(5) + '.h5'), "r")
   
    for i in range(1,200):
      rgb = data['images_center'][i]
      Image.fromarray(rgb).save(testPath + "rgb_" + str(i+h_num*200).zfill(5) + ".png")

