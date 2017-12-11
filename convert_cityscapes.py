import os, glob, sys, scipy
import numpy as np
from PIL import Image

dataset_dir = '/media/matthias/7E0CF8640CF818BB/Datasets/Cityscapes/'
#save_path_rgb = '/media/matthias/7E0CF8640CF818BB/Github/SemanticSegmentation/dataset/CityscapesCoarse/train/'
#save_path_seg = '/media/matthias/7E0CF8640CF818BB/Github/SemanticSegmentation/dataset/CityscapesCoarse/trainannot/'
save_path_rgb = '/media/matthias/7E0CF8640CF818BB/Github/SemanticSegmentation/dataset/RSSCityscapes2c/train/'
save_path_seg = '/media/matthias/7E0CF8640CF818BB/Github/SemanticSegmentation/dataset/RSSCityscapes2c/trainannot/'
dataset_name = 'train_extra'

if not os.path.exists(save_path_rgb):
    os.makedirs(save_path_rgb)
if not os.path.exists(save_path_seg):
    os.makedirs(save_path_seg)

#number_of_seg_classes = 5
#classes_join ={0:2,1:2,2:2,3:2,4:2,5:2,6:2,7:4,8:3,9:2,10:2,11:2,12:2,13:2,14:2,15:2,16:2,17:2,18:2,19:2,20:2,21:2,22:2,23:2,24:0,25:0,26:1,27:1,28:1,27:1,28:1,29:1,30:1,31:1,32:1,33:1}

number_of_seg_classes = 2
classes_join ={0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:1,8:0,9:0,10:0,11:0,12:0,13:0,14:0,15:0,16:0,17:0,18:0,19:0,20:0,21:0,22:0,23:0,24:0,25:0,26:0,27:0,28:0,27:0,28:0,29:0,30:0,31:0,32:0,33:0}

'''
unlabeled		0
ego vehicle		1
rectification border	2
out of roi		3
static			4
dynamic			5
ground 			6
road			7
sidewalk		8
parking			9
rail track		10
building		11
wall			12
fence			13
guard rail		14
bridge 			15
tunnel			16
pole			17
polegroup		18
traffic light		19
traffic sign		20
vegetation		21
terrain			22
sky			23
person			24
rider			25
car			26
truck			27
bus			28
caravan			29
trailer			30
train			31
motorcycle		32
bicycle			33
license plate		-1
'''

def join_classes(labels_image,join_dic):
  
  compressed_labels_image = np.copy(labels_image) 
  for key,value in join_dic.items():
    compressed_labels_image[np.where(labels_image==key)] = value
  return compressed_labels_image


# Where to look for Cityscapes
cityscapesPath = dataset_dir

# search for files
searchFineRGB   = os.path.join( cityscapesPath , "leftImg8bit", dataset_name , "*" ,"*.png" )
if dataset_name == 'train_extra':
	searchFineGT   = os.path.join( cityscapesPath , "gtCoarse"   , dataset_name , "*" ,"*labelIds.png" )
else:
	searchFineGT   = os.path.join( cityscapesPath , "gtFine"   , dataset_name , "*" ,"*labelIds.png" )

#searchCoarse = os.path.join( cityscapesPath , "gtCoarse" , dataset_name , "*" , "*.png" )

# search files
filesFineRGB = glob.glob( searchFineRGB )
filesFineRGB.sort()
filesFineGT = glob.glob( searchFineGT )
filesFineGT.sort()

print (len(filesFineRGB))

for i in range (len(filesFineRGB)):
	print('Progress: ', str(i+1), ' of ', str(len(filesFineRGB)))
	image = Image.open(filesFineRGB[i])
	image = image.crop((0, 0, 2048, 900))
	image = image.resize((200,88),resample = Image.BICUBIC)
	image.save(save_path_rgb + "cityscapes_rgb_" + dataset_name + str(i) + ".png")

	seg = Image.open(filesFineGT[i])
	seg = seg.crop((0, 0, 2048, 900))
	seg = seg.resize((200,88),resample = Image.NEAREST)
	seg_joined = join_classes(np.array(seg),classes_join)
	Image.fromarray(seg_joined).save(save_path_seg + "cityscapes_seg_" + dataset_name + str(i) + ".png")

