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

def get_one_hot_encoding(array,numLabels):
	mask = np.zeros((array.shape[0],array.shape[1],numLabels))
	for x in range(len(array)):
	    row = array[x]
	    for y in range(len(row)):
	        label = row[y]
		mask[x,y,label] = 1
	return mask


#Shrink to 5 classes
join_dic = {4:0,10:1,0:2,1:2,2:2,3:2,5:2,12:2,9:2,11:2,8:3,6:3,7:4}

def join_classes(labels_image,join_dic):
  
  compressed_labels_image = np.copy(labels_image) 
  for key,value in join_dic.iteritems():
    compressed_labels_image[np.where(labels_image==key)] = value


  return compressed_labels_image

# ***** main loop *****
if __name__ == "__main__":
  
  # Concatenate all files
  name = 'RCTest' #'CVPR25Noise'
  dataset_dir = './dataset/'
  #in_dir = '/media/matthias/7E0CF8640CF818BB/Github/ModularEnd2End/Desktop/CVPR25Noise/SeqTrain/'
  in_dir = '/media/matthias/7E0CF8640CF818BB/Datasets/RCTruck/8_Outdoors_4/'

  out_dir = dataset_dir + name + '/'


  trainPath = out_dir + 'train/'
  trainAnnotPath = out_dir + 'trainannot/'
  valPath = out_dir + 'val/'
  valAnnotPath = out_dir + 'valannot/'
  testPath = out_dir + 'test/'
  testAnnotPath = out_dir + 'testannot/'
  files = [os.path.join(in_dir, f) for f in glob.glob1(in_dir, "data_*.h5")]

  h5_start = 0
  h5_last = 13500
  bRGB = False
  bSeg = True
  number_of_seg_classes = 5 #13
  outdim = 1 

  if not os.path.exists(out_dir):
    os.mkdir(out_dir)

  if not os.path.exists(trainPath):
    os.mkdir(trainPath)
  if not os.path.exists(trainAnnotPath):
    os.mkdir(trainAnnotPath)

  if not os.path.exists(valPath):
    os.mkdir(valPath)
  if not os.path.exists(valAnnotPath):
    os.mkdir(valAnnotPath)

  if not os.path.exists(testPath):
    os.mkdir(testPath)
  if not os.path.exists(testAnnotPath):
    os.mkdir(testAnnotPath)

  # Now go over all files   
  sequence_num = range(h5_start,h5_last+1)
  for h_num in sequence_num:

    print " SEQUENCE NUMBER ",h_num
    data = h5py.File(os.path.join(in_dir,'data_' + str(h_num).zfill(5) + '.h5'), "r")
    #print(data['rgb'].shape)
    #print(data['labels'].shape)

   
    for i in range(1,200):
      rgb = data['images_center'][i]
      #rgb = data['rgb'][i]
      #if (outdim == 1):
        #scene_seg_raw = data['labels'][i][:,:,0] #*int(255/(number_of_seg_classes-1))
        #scene_seg = join_classes(scene_seg_raw,join_dic)
	#scene_seg = data['labels'][i][:,:,0]/int(255/(number_of_seg_classes-1))

      if (outdim == 3):
	scene_seg = np.zeros((100,200,3))
	scene_seg_hot = get_one_hot_encoding(data['labels'][i][:,:,0],number_of_seg_classes)
	#print (scene_seg_hot.shape)
	#0-boundary, 1-obstacles, 2-road
	for layer in range(scene_seg_hot.shape[2]):
		#if layer == 0: #sky
		#	scene_seg[:,:,0] += scene_seg_hot[:,:,layer]*255
		if layer == 1: #building
			scene_seg[:,:,0] += scene_seg_hot[:,:,layer]*255 
		if layer == 2: #fence
			scene_seg[:,:,0] += scene_seg_hot[:,:,layer]*255 
		#if layer == 3: #misc
		#	scene_seg[:,:,0] += scene_seg_hot[:,:,layer]*255 
		if layer == 4: #pedestrian
			scene_seg[:,:,1] += scene_seg_hot[:,:,layer]*255 
		if layer == 5: #pole
			scene_seg[:,:,0] += scene_seg_hot[:,:,layer]*255 
		if layer == 6: #landmarking
			scene_seg[:,:,0] += scene_seg_hot[:,:,layer]*255 
		if layer == 7: #road
			scene_seg[:,:,2] += scene_seg_hot[:,:,layer]*255 
		if layer == 8: #sidewalk
			scene_seg[:,:,0] += scene_seg_hot[:,:,layer]*255 
		if layer == 9: #vegetation
			scene_seg[:,:,0] += scene_seg_hot[:,:,layer]*255 
		if layer == 10: #vehicles
			scene_seg[:,:,1] += scene_seg_hot[:,:,layer]*255 
		if layer == 11: #wall
			scene_seg[:,:,0] += scene_seg_hot[:,:,layer]*255 
		if layer == 12: #traffic sign
			scene_seg[:,:,0] += scene_seg_hot[:,:,layer]*255 

      #rand = random.random()
      #if (outdim == 3):
      	#Image.fromarray(segs_center[i]).convert('RGB').save("seg_" + str(i) + ".png")
      '''
      if (outdim == 1):
	if h_num < h5_last*0.2:
	  Image.fromarray(rgb).save(testPath + name + "_rgb_" + str(i+h_num*200) + "_W8.png")
      	  Image.fromarray(scene_seg).save(testAnnotPath + name + "_seg_" + str(i+h_num*200) + "_W8.png") 
	elif h_num < h5_last*0.4:
	  Image.fromarray(rgb).save(valPath + name + "_rgb_" + str(i+h_num*200) + "_W8.png")
      	  Image.fromarray(scene_seg).save(valAnnotPath + name + "_seg_" + str(i+h_num*200) + "_W8.png") 
	else:
	  Image.fromarray(rgb).save(trainPath + name + "_rgb_" + str(i+h_num*200) + "_W8.png")
      	  Image.fromarray(scene_seg).save(trainAnnotPath + name + "_seg_" + str(i+h_num*200) + "_W8.png") 
      '''
      Image.fromarray(rgb).save(trainPath + name + "_rgb_" + str(i+h_num*200) + ".png")
      #Image.fromarray(scene_seg).save(trainAnnotPath + name + "_seg_" + str(i+h_num*200) + ".png") 
      #Image.fromarray(rgb).save(valPath + name + "_rgb_" + str(i+h_num*200) + ".png")
      #Image.fromarray(scene_seg).save(valAnnotPath + name + "_seg_" + str(i+h_num*200) + ".png") 
