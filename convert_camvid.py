import os
import scipy
import numpy as np
from PIL import Image

dataset_dir = '/media/matthias/7E0CF8640CF818BB/Datasets/CamVid/'
#save_path_rgb = '/media/matthias/7E0CF8640CF818BB/Github/SemanticSegmentation/dataset/CamVid/train/'
#save_path_seg = '/media/matthias/7E0CF8640CF818BB/Github/SemanticSegmentation/dataset/CamVid/trainannot/'
save_path_rgb = '/media/matthias/7E0CF8640CF818BB/Github/SemanticSegmentation/dataset/RSSCamVid2c/train/'
save_path_seg = '/media/matthias/7E0CF8640CF818BB/Github/SemanticSegmentation/dataset/RSSCamVid2c/trainannot/'
dataset_name = 'train'

if not os.path.exists(save_path_rgb):
	os.mkdir(save_path_rgb)
if not os.path.exists(save_path_seg):
	os.mkdir(save_path_seg)

number_of_seg_classes = 2
classes_join ={0:0,1:0,2:0,3:1,4:0,5:0,6:0,7:0,8:0,9:0,10:0,11:0,12:0}

'''
0  - Sky = [128,128,128] ?
1  - Building = [128,0,0] ?
2  - Pole = [192,192,128] ?
3  - Road_marking = [255,69,0]
4  - Sidewalk = [128,64,128]
5  - Pavement = [60,40,222] ?
6  - Tree = [128,128,0] ?
7  - SignSymbol = [192,128,128] ?
8  - Fence = [64,64,128] ?
9  - Car = [64,0,128] ?
10 - Pedestrian = [64,64,0] ?
11 - Bicyclist = [0,128,192] ?
12 - Unlabelled = [0,0,0] ?
'''

def join_classes(labels_image,join_dic):
  
  compressed_labels_image = np.copy(labels_image) 
  for key,value in join_dic.items():
    compressed_labels_image[np.where(labels_image==key)] = value
  return compressed_labels_image

#Dataset directories
image_files = sorted([os.path.join(dataset_dir, dataset_name, 'raw_images', file) for file in os.listdir(os.path.join(dataset_dir, dataset_name, 'raw_images')) if file.endswith('.png')])

annotation_files = sorted([os.path.join(dataset_dir, dataset_name, "class_id", file) for file in os.listdir(os.path.join(dataset_dir, dataset_name, "class_id")) if file.endswith('.png')])

print (len(image_files))

for i in range (len(image_files)):
	print('Progress: ', str(i+1), ' of ', str(len(image_files)))
	image = Image.open(image_files[i])
	image = image.crop((0, 150, 480, 360))
	image = image.resize((200,88),resample = Image.BICUBIC)
	image.save(save_path_rgb + "camvid_rgb_" + dataset_name + str(i) + ".png")

	seg = Image.open(annotation_files[i])
	seg = seg.crop((0, 150, 480, 360))
	seg = seg.resize((200,88),resample = Image.NEAREST)
	seg_joined = join_classes(np.array(seg),classes_join)
	Image.fromarray(seg_joined).save(save_path_seg + "camvid_seg_" + dataset_name + str(i) + ".png")

