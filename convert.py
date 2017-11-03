import tensorflow as tf
import os
from PIL import Image
import numpy as np
import skimage.io as io

#==============INPUT ARGUMENTS==================
flags = tf.app.flags

#Directory arguments
flags.DEFINE_string('dataset_dir', './dataset', 'The dataset base directory.')
flags.DEFINE_string('dataset_name', 'CVPR1Noise', 'The dataset subdirectory to find the train, validation and test images.')

FLAGS = flags.FLAGS

dataset_dir = FLAGS.dataset_dir
dataset_name = FLAGS.dataset_name

#===============PREPARATION FOR TRAINING==================
#Get the images into a list
image_files = sorted([os.path.join(dataset_dir, dataset_name, 'train', file) for file in os.listdir(os.path.join(dataset_dir, dataset_name, 'train')) if file.endswith('.png')])
annotation_files = sorted([os.path.join(dataset_dir, dataset_name, 'trainannot', file) for file in os.listdir(os.path.join(dataset_dir, dataset_name, 'trainannot')) if file.endswith('.png')])

image_val_files = sorted([os.path.join(dataset_dir,'CVPRVal', 'val', file) for file in os.listdir(os.path.join(dataset_dir,'CVPRVal', 'val')) if file.endswith('.png')])
annotation_val_files = sorted([os.path.join(dataset_dir, 'CVPRVal', 'valannot', file) for file in os.listdir(os.path.join(dataset_dir, 'CVPRVal', 'valannot')) if file.endswith('.png')])

num_files_train = len(image_files)

print ('Number of images: ' , len(image_files))
print ('First image path: ' , image_files[0])
print ('Last image path: ' , image_files[num_files_train-1])
print ('First anno path: ' , annotation_files[0])
print ('Last anno path: ' , annotation_files[num_files_train-1])


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

tfrecords_filename_train = dataset_dir + '_train.tfrecords'
tfrecords_filename_val = dataset_dir + '_val.tfrecords'


writer_train = tf.python_io.TFRecordWriter(tfrecords_filename_train)

# Let's collect the real images to later on compare
# to the reconstructed ones
original_images = []
check_num = 50

for i in range(num_files_train):
    if (i%1000 == 0):
      print ('Progress: %d of %d (%.f%%)' % (i, num_files_train, float(i*100/num_files_train)))
    img_path = image_files[i]
    annotation_path = annotation_files[i]
    img = np.array(Image.open(img_path))
    annotation = np.array(Image.open(annotation_path))
    
    # The reason to store image sizes was demonstrated
    # in the previous example -- we have to know sizes
    # of images to later read raw serialized string,
    # convert to 1d array and convert to respective
    # shape that image used to have.
    height = img.shape[0]
    width = img.shape[1]
    
    # Put in the original images into array
    # Just for future check for correctness
    if i<check_num:
    	original_images.append((img, annotation))
    
    img_raw = img.tostring()
    annotation_raw = annotation.tostring()
    
    example = tf.train.Example(features=tf.train.Features(feature={
        'height': _int64_feature(height),
        'width': _int64_feature(width),
        'image_raw': _bytes_feature(img_raw),
        'mask_raw': _bytes_feature(annotation_raw)}))
    
    writer_train.write(example.SerializeToString())

writer_train.close()


###Sanity check - Are the images identical?
reconstructed_images = []
i = 0

record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename_train)

for string_record in record_iterator:

        
    example = tf.train.Example()
    example.ParseFromString(string_record)
    
    height = int(example.features.feature['height']
                                 .int64_list
                                 .value[0])
    
    width = int(example.features.feature['width']
                                .int64_list
                                .value[0])
    
    img_string = (example.features.feature['image_raw']
                                  .bytes_list
                                  .value[0])
    
    annotation_string = (example.features.feature['mask_raw']
                                .bytes_list
                                .value[0])
    
    img_1d = np.fromstring(img_string, dtype=np.uint8)
    reconstructed_img = img_1d.reshape((height, width, -1))
    
    annotation_1d = np.fromstring(annotation_string, dtype=np.uint8)
    
    # Annotations don't have depth (3rd dimension)
    reconstructed_annotation = annotation_1d.reshape((height, width))
    
    reconstructed_images.append((reconstructed_img, reconstructed_annotation))

    i = i + 1
    if i == check_num:
        break

# Let's check if the reconstructed images match
# the original images

for original_pair, reconstructed_pair in zip(original_images, reconstructed_images):
    
    img_pair_to_compare, annotation_pair_to_compare = zip(original_pair,
                                                          reconstructed_pair)
    print('Images identical: ', np.allclose(*img_pair_to_compare))
    print('Annotations identical: ', np.allclose(*annotation_pair_to_compare))

###Validation
num_files_val = len(image_val_files)

print ('Number of images: ' , len(image_val_files))
print ('First image path: ' , image_val_files[0])
print ('Last image path: ' , image_val_files[num_files_val-1])
print ('First anno path: ' , annotation_val_files[0])
print ('Last anno path: ' , annotation_val_files[num_files_val-1])
writer_val = tf.python_io.TFRecordWriter(tfrecords_filename_val)

for i in range(num_files_val):
    if (i%1000 == 0):
      print ('Progress: %d of %d (%.f%%)' % (i, num_files_val, float(i*100/num_files_val)))
    img_path = image_val_files[i]
    annotation_path = annotation_val_files[i]
    img = np.array(Image.open(img_path))
    annotation = np.array(Image.open(annotation_path))
    
    # The reason to store image sizes was demonstrated
    # in the previous example -- we have to know sizes
    # of images to later read raw serialized string,
    # convert to 1d array and convert to respective
    # shape that image used to have.
    height = img.shape[0]
    width = img.shape[1]
    
   
    img_raw = img.tostring()
    annotation_raw = annotation.tostring()
    
    example = tf.train.Example(features=tf.train.Features(feature={
        'height': _int64_feature(height),
        'width': _int64_feature(width),
        'image_raw': _bytes_feature(img_raw),
        'mask_raw': _bytes_feature(annotation_raw)}))
    
    writer_val.write(example.SerializeToString())

writer_val.close()

