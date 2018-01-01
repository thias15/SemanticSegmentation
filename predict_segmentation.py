import tensorflow as tf
import os
import matplotlib.pyplot as plt
from enet import ENet, ENet_arg_scope, ENet_Small
from erfnet import ErfNet, ErfNet_Small
from preprocessing import preprocess
from scipy.misc import imsave,imread
import numpy as np
slim = tf.contrib.slim

#image_dir = './dataset/RCTest/test/'
image_dir = './dataset/RCSeg/test/'
images_list = sorted([os.path.join(image_dir, file) for file in os.listdir(image_dir) if file.endswith('.png')])

checkpoint_dir = "./checkpoint"
checkpoint = tf.train.latest_checkpoint(checkpoint_dir)

network = 'ErfNet_Small'
num_initial_blocks = 1
skip_connections = False
stage_two_repeat = 1
batch_size = 1 #10
num_classes = 2 #12
image_height = 88
image_width = 200
is_training = False

'''
#Labels to colours are obtained from here:
https://github.com/alexgkendall/SegNet-Tutorial/blob/c922cc4a4fcc7ce279dd998fb2d4a8703f34ebd7/Scripts/test_segmentation_camvid.py

However, the road_marking class is collapsed into the road class in the dataset provided.

Classes - CamVid:
------------
Sky = [128,128,128]
Building = [128,0,0]
Pole = [192,192,128]
Road_marking = [255,69,0]
Road = [128,64,128]
Pavement = [60,40,222]
Tree = [128,128,0]
SignSymbol = [192,128,128]
Fence = [64,64,128]
Car = [64,0,128]
Pedestrian = [64,64,0]
Bicyclist = [0,128,192]
Unlabelled = [0,0,0]

Classes - Carla_5:
------------
Pedestrian = [220, 20, 60] - red
Car = [0,  0, 142] - blue
Obstacle = [255, 255, 0] - yellow
Sidewalk = [0, 255, 255] - cyan 
Road = [128, 64, 128] - purple 

Classes - Carla_2:
------------
Background = [255, 255, 0] - yellow
Road = [128, 64, 128] - purple 
'''

encoding_camvid = {0: [128,128,128],
                     1: [128,0,0],
                     2: [192,192,128],
                     3: [128,64,128],
                     4: [60,40,222],
                     5: [128,128,0],
                     6: [192,128,128],
                     7: [64,64,128],
                     8: [64,0,128],
                     9: [64,64,0],
                     10: [0,128,192],
                     11: [0,0,0]}

encoding_carla_5 = {0: [220, 20, 60],
                     1: [0,  0, 142],
                     2: [255, 255, 0],
                     3: [0, 255, 255],
                     4: [128, 64, 128]}

encoding_carla_2 = {0: [255, 255, 0],
                    1: [128, 64, 128]}

label_to_colours =  encoding_carla_2

#Create the photo directory
photo_dir = checkpoint_dir + "/test_images"
if not os.path.exists(photo_dir):
    os.mkdir(photo_dir)

#Create a function to convert each pixel label to colour.
def grayscale_to_colour(image):
    print 'Converting image...'
    image = image.reshape((image_height, image_width, 1))
    image = np.repeat(image, 3, axis=-1)
    for i in xrange(image.shape[0]):
        for j in xrange(image.shape[1]):
            label = int(image[i][j][0])
            image[i][j] = np.array(label_to_colours[label])

    return image


with tf.Graph().as_default() as graph:
    images_tensor = tf.train.string_input_producer(images_list, shuffle=False)
    reader = tf.WholeFileReader()
    key, image_tensor = reader.read(images_tensor)
    image = tf.image.decode_png(image_tensor, channels=3)
    # image = tf.image.resize_image_with_crop_or_pad(image, 360, 480)
    # image = tf.cast(image, tf.float32)
    image = preprocess(image,height=image_height, width=image_width)
    images = tf.train.batch([image], batch_size = batch_size, allow_smaller_final_batch=True)

    #Create the model inference
    with slim.arg_scope(ENet_arg_scope()):
        if (network == 'ENet'):
	    print ('Building the network: ' , network)
	    logits, probabilities = ENet(images,
	                         num_classes,
	                         batch_size=batch_size,
	                         is_training=is_training,
	                         reuse=None,
	                         num_initial_blocks=num_initial_blocks,
	                         stage_two_repeat=stage_two_repeat,
	                         skip_connections=skip_connections)

        if (network == 'ENet_Small'):
	    print ('Building the network: ' , network)
	    logits, probabilities = ENet_Small(images,
	                         num_classes,
	                         batch_size=batch_size,
	                         is_training=is_training,
	                         reuse=None,
	                         num_initial_blocks=num_initial_blocks,
	                         skip_connections=skip_connections)

        if (network == 'ErfNet'):
	    print ('Building the network: ' , network)
	    logits, probabilities = ErfNet(images,
	                         num_classes,
	                         batch_size=batch_size,
	                         is_training=is_training,
	                         reuse=None)

        if (network == 'ErfNet_Small'):
	    print ('Building the network: ' , network)
	    logits, probabilities = ErfNet_Small(images,
	                         num_classes,
	                         batch_size=batch_size,
	                         is_training=is_training,
	                         reuse=None)

    variables_to_restore = slim.get_variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)
    def restore_fn(sess):
        return saver.restore(sess, checkpoint)

    predictions = tf.argmax(probabilities, -1)
    predictions = tf.cast(predictions, tf.float32)
    print 'HERE', predictions.get_shape()

    sv = tf.train.Supervisor(logdir=None, init_fn=restore_fn)
    
    with sv.managed_session() as sess:

        for i in xrange(len(images_list) / batch_size):
            segmentations = sess.run(predictions)
            # print segmentations.shape

            for j in xrange(segmentations.shape[0]):

		img = imread(images_list[i])
                colored_class_image = grayscale_to_colour(segmentations[j])
		alpha_blended = 0.5 * colored_class_image + 0.5 * img
		filename = photo_dir + "/image_%s" %(i*batch_size + j)
		ext = '.png'
		#imsave(filename + "_seg" + ext, colored_class_image)
		imsave(filename + "_seg_blended" + ext, alpha_blended)


		print 'Saving image %s/%s' %(i*batch_size + j, len(images_list))
                #plt.axis('off')
                #plt.imshow(converted_image)
                #imsave(photo_dir + "/image_%s.png" %(i*batch_size + j), converted_image)
                # plt.show()
