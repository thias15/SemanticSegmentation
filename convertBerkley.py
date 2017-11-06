import os

dataset_dir = '/home/matthias/Downloads/segmentation/'
dataset_name 'train'

image_cut =[60,700]
number_of_seg_classes = 5
classes_join ={0:2,1:2,2:2,3:2,4:2,5:4,6:4,7:4,8:3,9:2,10:2,11:2,12:2,13:2,14:2,15:2,16:2,17:2,18:3,19:2,20:2,21:2,22:2,23:2,24:2,25:2,26:2,27:2,28:2,27:2,28:2,29:2,30:2,31:0,32:0,33:1,34:1,35:1,36:1,37:1,38:1,39:1,40:1}
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
  for key,value in join_dic.iteritems():
    compressed_labels_image[np.where(labels_image==key)] = value


  return compressed_labels_image

#Dataset directories
image_files = sorted([os.path.join(dataset_dir, dataset_name, 'raw_images', file) for file in os.listdir(os.path.join(dataset_dir, dataset_name, 'raw_images')) if file.endswith('.jpg')])

annotation_files = sorted([os.path.join(dataset_dir, dataset_name, "class_id", file) for file in os.listdir(os.path.join(dataset_dir, dataset_name, "class_id")) if file.endswith('.png')])



for

image = image_files[i][self._image_cut[0]:self._image_cut[1],:,:3]

join_classes(annotation,classes_join)
