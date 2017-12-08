import os
import scipy
import numpy as np
from PIL import Image

dataset_dir = '/media/matthias/7E0CF8640CF818BB/Datasets/Berkeley/'
#save_path_rgb = '/media/matthias/7E0CF8640CF818BB/Github/SemanticSegmentation/dataset/Berkeley/train/'
#save_path_seg = '/media/matthias/7E0CF8640CF818BB/Github/SemanticSegmentation/dataset/Berkeley/trainannot/'
save_path_rgb = '/media/matthias/7E0CF8640CF818BB/Github/SemanticSegmentation/dataset/RSSBerkeley2c/train/'
save_path_seg = '/media/matthias/7E0CF8640CF818BB/Github/SemanticSegmentation/dataset/RSSBerkeley2c/trainannot/'
dataset_name = 'val'

if not os.path.exists(save_path_rgb):
	os.mkdir(save_path_rgb)
if not os.path.exists(save_path_seg):
	os.mkdir(save_path_seg)

#number_of_seg_classes = 5
#classes_join ={0:2,1:2,2:2,3:2,4:2,5:4,6:4,7:4,8:3,9:2,10:2,11:2,12:2,13:2,14:2,15:2,16:2,17:2,18:3,19:2,20:2,21:2,22:2,23:2,24:2,25:2,26:2,27:2,28:2,27:2,28:2,29:2,30:2,31:0,32:0,33:1,34:1,35:1,36:1,37:1,38:1,39:1,40:1}

number_of_seg_classes = 2
classes_join ={0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:1,8:0,9:0,10:0,11:0,12:0,13:0,14:0,15:0,16:0,17:0,18:1,19:0,20:0,21:0,22:0,23:0,24:0,25:0,26:0,27:0,28:0,27:0,28:0,29:0,30:0,31:0,32:0,33:0,34:0,35:0,36:0,37:0,38:0,39:0,40:0}

'''
unlabeled		0
dynamic			1
ego vehicle		2
ground			3
static			4
parking			5
rail track		6
road			7
sidewalk		8
bridge			9
building		10
fence			11
garage			12
guard rail		13
tunnel			14
wall 			15
banner			16
billboard		17
lane divider		18
parking sign		19
pole			20
polegroup		21
street light		22
traffic cone		23
traffic device		24
traffic light		25
traffic sign		26
traffic sign frame	27
terrain			28
vegetation		29
sky			30
person			31
rider			32
bicycle			33
bus			34
car			35
caravan			36
motorcycle		37
trailer			38
train			39
truck			40
'''

def join_classes(labels_image,join_dic):
  
  compressed_labels_image = np.copy(labels_image) 
  for key,value in join_dic.items():
    compressed_labels_image[np.where(labels_image==key)] = value
  return compressed_labels_image

#Dataset directories
image_files = sorted([os.path.join(dataset_dir, dataset_name, 'raw_images', file) for file in os.listdir(os.path.join(dataset_dir, dataset_name, 'raw_images')) if file.endswith('.jpg')])

annotation_files = sorted([os.path.join(dataset_dir, dataset_name, "class_id", file) for file in os.listdir(os.path.join(dataset_dir, dataset_name, "class_id")) if file.endswith('.png')])

print (len(image_files))

for i in range (len(image_files)):
	print('Progress: ', str(i+1), ' of ', str(len(image_files)))
	image = Image.open(image_files[i])
	image = image.crop((0, 150, 1280, 710))
	image = image.resize((200,88),resample = Image.BICUBIC)
	image.save(save_path_rgb + "berkley_rgb_" + dataset_name + str(i) + ".png")

	seg = Image.open(annotation_files[i])
	seg = seg.crop((0, 160, 1280, 720))
	seg = seg.resize((200,88),resample = Image.NEAREST)
	seg_joined = join_classes(np.array(seg),classes_join)
	Image.fromarray(seg_joined).save(save_path_seg + "berkley_seg_" + dataset_name + str(i) + ".png")

