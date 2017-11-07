import tensorflow as tf
import os
from PIL import Image
import numpy as np
import skimage.io as io

tfrecords_filename = './dataset/CVPR1Noise_train.tfrecords'

record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)

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
    Image.fromarray(reconstructed_img).save('rgb.png')

    annotation_1d = np.fromstring(annotation_string, dtype=np.uint8)
    # Annotations don't have depth (3rd dimension)
    reconstructed_annotation = annotation_1d.reshape((height, width))
    Image.fromarray(reconstructed_annotation).save('anno.png')
    break
