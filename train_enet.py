import tensorflow as tf
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from tensorflow.python.platform import tf_logging as logging
from enet import ENet, ENet_arg_scope, ENet_Small
from erfnet import ErfNet, ErfNet_Small, ErfNet_NoDS
from preprocessing import preprocess
from get_class_weights import ENet_weighing, median_frequency_balancing, CVPR_weighing
import os
import time
import numpy as np
import matplotlib.pyplot as plt
slim = tf.contrib.slim

#==============INPUT ARGUMENTS==================
flags = tf.app.flags

#Directory arguments
flags.DEFINE_string('dataset_dir', './dataset', 'The dataset base directory.')
flags.DEFINE_string('dataset_name', 'Carla', 'The dataset subdirectory to find the train images.')
flags.DEFINE_string('validation_name', 'CVPRVal', 'The dataset subdirectory to find validation images.')
flags.DEFINE_string('logdir', './log', 'The log directory to save your checkpoint and event files.')
flags.DEFINE_boolean('save_images', True, 'Whether or not to save your images.')
flags.DEFINE_boolean('combine_dataset', False, 'If True, combines the validation with the train dataset.')

#Training arguments
flags.DEFINE_string('network', 'ENet_Small', 'The type of network to use.') 
flags.DEFINE_integer('num_classes', 5, 'The number of classes to predict.') #12
flags.DEFINE_integer('batch_size', 10, 'The batch_size for training.') #10
flags.DEFINE_integer('eval_batch_size', 20, 'The batch size used for validation.') #25
flags.DEFINE_integer('image_height', 88, "The input height of the images.") #360
flags.DEFINE_integer('image_width', 200, "The input width of the images.") #480
flags.DEFINE_integer('num_epochs', 100, "The number of epochs to train your model.")
flags.DEFINE_integer('num_epochs_before_decay', 10, 'The number of epochs before decaying your learning rate.') #100
flags.DEFINE_integer('decay_steps', 0, 'The number of steps before decaying your learning rate.') 
flags.DEFINE_float('weight_decay', 2e-4, "The weight decay for ENet convolution layers.")
flags.DEFINE_float('learning_rate_decay_factor', 1e-1, 'The learning rate decay factor.')
flags.DEFINE_float('initial_learning_rate', 5e-5, 'The initial learning rate for your training.') #5e-4
flags.DEFINE_string('weighting', "MFB", 'Choice of Median Frequency Balancing or the custom ENet class weights.')
flags.DEFINE_string('checkpoint_step', 10000, 'Number of steps between checkpoints.')
flags.DEFINE_string('log_step', 100, 'Number of steps between logs.')
flags.DEFINE_string('print_step', 50, 'Number of steps between prints.')
flags.DEFINE_string('val_step', 100, 'Number of steps between validations.')

#Architectural changes
flags.DEFINE_integer('num_initial_blocks', 1, 'The number of initial blocks to use in ENet.')
flags.DEFINE_integer('stage_two_repeat', 1, 'The number of times to repeat stage two.')
flags.DEFINE_boolean('skip_connections', False, 'If True, perform skip connections from encoder to decoder.')

FLAGS = flags.FLAGS

#==========NAME HANDLING FOR CONVENIENCE==============
network = FLAGS.network
num_classes = FLAGS.num_classes
batch_size = FLAGS.batch_size
image_height = FLAGS.image_height
image_width = FLAGS.image_width
eval_batch_size = FLAGS.eval_batch_size #Can be larger than train_batch as no need to backpropagate gradients.
combine_dataset = FLAGS.combine_dataset

checkpoint_step = FLAGS.checkpoint_step
log_step = FLAGS.log_step
print_step = FLAGS.print_step
val_step = FLAGS.val_step

#Training parameters
initial_learning_rate = FLAGS.initial_learning_rate
num_epochs_before_decay = FLAGS.num_epochs_before_decay
num_epochs =FLAGS.num_epochs
learning_rate_decay_factor = FLAGS.learning_rate_decay_factor
weight_decay = FLAGS.weight_decay
epsilon = 1e-8

#Architectural changes
num_initial_blocks = FLAGS.num_initial_blocks
stage_two_repeat = FLAGS.stage_two_repeat
skip_connections = FLAGS.skip_connections

#Use median frequency balancing or not
weighting = FLAGS.weighting

#Visualization and where to save images
save_images = FLAGS.save_images

#Directories
dataset_dir = FLAGS.dataset_dir
dataset_name = FLAGS.dataset_name 
validation_name = FLAGS.validation_name
logdir = os.path.join(FLAGS.logdir,'train_' + FLAGS.dataset_name + '_' + FLAGS.network + '_' + FLAGS.weighting + '_lr_' + str(FLAGS.initial_learning_rate) + '_bs_' + str(FLAGS.batch_size))

photo_dir = os.path.join(logdir, "images")

print('Dataset: ',dataset_name)
print('Validation: ',validation_name)

#===============PREPARATION FOR TRAINING==================
#Get the images into a list
image_files = sorted([os.path.join(dataset_dir, dataset_name, 'train', file) for file in os.listdir(os.path.join(dataset_dir, dataset_name, 'train')) if file.endswith('.png')])
annotation_files = sorted([os.path.join(dataset_dir, dataset_name, 'trainannot', file) for file in os.listdir(os.path.join(dataset_dir, dataset_name, 'trainannot')) if file.endswith('.png')])

image_val_files = sorted([os.path.join(dataset_dir, validation_name, 'val', file) for file in os.listdir(os.path.join(dataset_dir, validation_name, 'val')) if file.endswith('.png')])
annotation_val_files = sorted([os.path.join(dataset_dir, validation_name, 'valannot', file) for file in os.listdir(os.path.join(dataset_dir, validation_name, 'valannot')) if file.endswith('.png')])

if combine_dataset:
    image_files += image_val_files
    annotation_files += annotation_val_files

#Know the number steps to take before decaying the learning rate and batches per epoch
num_batches_per_epoch = len(image_files) / batch_size
num_steps_per_epoch = num_batches_per_epoch

if (FLAGS.decay_steps > 0):
    decay_steps = FLAGS.decay_steps
else:
    decay_steps = int(num_epochs_before_decay * num_steps_per_epoch)


#=================CLASS WEIGHTS===============================
#Median frequency balancing class_weights
if weighting == "MFB":
    class_weights = median_frequency_balancing(image_dir = os.path.join(dataset_dir, dataset_name, 'trainannot'), num_classes=num_classes)
    print "========= Median Frequency Balancing Class Weights =========\n", class_weights

#Inverse weighing probability class weights
elif weighting == "ENET":
    class_weights = ENet_weighing(image_dir = os.path.join(dataset_dir, dataset_name, 'trainannot'), num_classes=num_classes)
    print "========= ENet Class Weights =========\n", class_weights

#Inverse weighing probability class weights
elif weighting == "CVPR":
    class_weights = CVPR_weighing()
    print "========= CVPR Class Weights =========\n", class_weights

#============= TRAINING =================
def weighted_cross_entropy(onehot_labels, logits, class_weights):
    '''
    A quick wrapper to compute weighted cross entropy. 

    ------------------
    Technical Details
    ------------------
    The class_weights list can be multiplied by onehot_labels directly because the last dimension
    of onehot_labels is 12 and class_weights (length 12) can broadcast across that dimension, which is what we want. 
    Then we collapse the last dimension for the class_weights to get a shape of (batch_size, height, width, 1)
    to get a mask with each pixel's value representing the class_weight.

    This mask can then be that can be broadcasted to the intermediate output of logits
    and onehot_labels when calculating the cross entropy loss.
    ------------------

    INPUTS:
    - onehot_labels(Tensor): the one-hot encoded labels of shape (batch_size, height, width, num_classes)
    - logits(Tensor): the logits output from the model that is of shape (batch_size, height, width, num_classes)
    - class_weights(list): A list where each index is the class label and the value of the index is the class weight.

    OUTPUTS:
    - loss(Tensor): a scalar Tensor that is the weighted cross entropy loss output.
    '''
    weights = onehot_labels * class_weights
    weights = tf.reduce_sum(weights, 3)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits, weights=weights)

    return loss

def run():
    with tf.Graph().as_default() as graph:
        tf.logging.set_verbosity(tf.logging.INFO)

        #===================TRAINING BRANCH=======================
        #Load the files into one input queue
        images = tf.convert_to_tensor(image_files)
        annotations = tf.convert_to_tensor(annotation_files)
        input_queue = tf.train.slice_input_producer([images, annotations]) #Slice_input producer shuffles the data by default.

        #Decode the image and annotation raw content
        image = tf.read_file(input_queue[0])
        image = tf.image.decode_image(image, channels=3)
        annotation = tf.read_file(input_queue[1])
        annotation = tf.image.decode_image(annotation)

        #preprocess and batch up the image and annotation
        preprocessed_image, preprocessed_annotation = preprocess(image, annotation, image_height, image_width)
        images, annotations = tf.train.batch([preprocessed_image, preprocessed_annotation], batch_size=batch_size, allow_smaller_final_batch=True)

        #Create the model inference
        with slim.arg_scope(ENet_arg_scope(weight_decay=weight_decay)):

	    if (network == 'ENet'):
		print ('Building the network: ' , network)
                logits, probabilities = ENet(images,
                                         num_classes,
                                         batch_size=batch_size,
                                         is_training=True,
                                         reuse=None,
                                         num_initial_blocks=num_initial_blocks,
                                         stage_two_repeat=stage_two_repeat,
                                         skip_connections=skip_connections)

	    if (network == 'ENet_Small'):
		print ('Building the network: ' , network)
                logits, probabilities = ENet_Small(images,
                                         num_classes,
                                         batch_size=batch_size,
                                         is_training=True,
                                         reuse=None,
                                         num_initial_blocks=num_initial_blocks,
                                         skip_connections=skip_connections)

	    if (network == 'ErfNet'):
		print ('Building the network: ' , network)
                logits, probabilities = ErfNet(images,
                                         num_classes,
                                         batch_size=batch_size,
                                         is_training=True,
                                         reuse=None)

	    if (network == 'ErfNet_Small'):
		print ('Building the network: ' , network)
                logits, probabilities = ErfNet_Small(images,
                                         num_classes,
                                         batch_size=batch_size,
                                         is_training=True,
                                         reuse=None)

	    if (network == 'ErfNet_NoDS'):
		print ('Building the network: ' , network)
                logits, probabilities = ErfNet_NoDS(images,
                                         num_classes,
                                         batch_size=batch_size,
                                         is_training=True,
                                         reuse=None)


        #perform one-hot-encoding on the ground truth annotation to get same shape as the logits
        annotations = tf.reshape(annotations, shape=[batch_size, image_height, image_width])
        annotations_ohe = tf.one_hot(annotations, num_classes, axis=-1)

        #Actually compute the loss
        loss = weighted_cross_entropy(logits=logits, onehot_labels=annotations_ohe, class_weights=class_weights)
        total_loss = tf.losses.get_total_loss()

        #Create the global step for monitoring the learning_rate and training.
        global_step = get_or_create_global_step()

        #Define your exponentially decaying learning rate
        lr = tf.train.exponential_decay(
            learning_rate = initial_learning_rate,
            global_step = global_step,
            decay_steps = decay_steps,
            decay_rate = learning_rate_decay_factor,
            staircase = True)

        #Now we can define the optimizer that takes on the learning rate
        optimizer = tf.train.AdamOptimizer(learning_rate=lr, epsilon=epsilon)

        #Create the train_op.
        train_op = slim.learning.create_train_op(total_loss, optimizer)

        #State the metrics that you want to predict. We get a predictions that is not one_hot_encoded.
        predictions = tf.argmax(probabilities, -1)
        accuracy, accuracy_update = tf.contrib.metrics.streaming_accuracy(predictions, annotations)
        mean_IOU, mean_IOU_update = tf.contrib.metrics.streaming_mean_iou(predictions=predictions, labels=annotations, num_classes=num_classes)

	weights = annotations_ohe * class_weights
	weights = tf.reduce_sum(weights, 3)
        mean_iIOU, mean_iIOU_update = tf.contrib.metrics.streaming_mean_iou(predictions=predictions, labels=annotations, num_classes=num_classes, weights = weights)
        metrics_op = tf.group(accuracy_update, mean_IOU_update,mean_iIOU_update)

        #Now we need to create a training step function that runs both the train_op, metrics_op and updates the global_step concurrently.
        def train_step(sess, train_op, global_step, metrics_op, print_step):
            '''
            Simply runs a session for the three arguments provided and gives a logging on the time elapsed for each global step
            '''
            #Check the time for each sess run
            start_time = time.time()
            total_loss, global_step_count, accuracy_value, mean_IOU_value, mean_iIOU_value, _ = sess.run([train_op, global_step, accuracy, mean_IOU, mean_iIOU, metrics_op])
            time_elapsed = time.time() - start_time

            #Run the logging to show some results
	    if (global_step_count % print_step == 0):
                logging.info('global step %s: loss: %.4f (%.2f sec/step)    Accuracy: %.4f    Mean IOU: %.4f    Mean iIOU: %.4f', global_step_count, total_loss, time_elapsed, accuracy_value, mean_IOU_value, mean_iIOU_value)

            return total_loss, accuracy_value, mean_IOU_value, mean_iIOU_value

        #================VALIDATION BRANCH========================
        #Load the files into one input queue
        images_val = tf.convert_to_tensor(image_val_files)
        annotations_val = tf.convert_to_tensor(annotation_val_files)
        input_queue_val = tf.train.slice_input_producer([images_val, annotations_val])

        #Decode the image and annotation raw content
        image_val = tf.read_file(input_queue_val[0])
        image_val = tf.image.decode_jpeg(image_val, channels=3)
        annotation_val = tf.read_file(input_queue_val[1])
        annotation_val = tf.image.decode_png(annotation_val)

        #preprocess and batch up the image and annotation
        preprocessed_image_val, preprocessed_annotation_val = preprocess(image_val, annotation_val, image_height, image_width)
        images_val, annotations_val = tf.train.batch([preprocessed_image_val, preprocessed_annotation_val], batch_size=eval_batch_size, allow_smaller_final_batch=True)

        with slim.arg_scope(ENet_arg_scope(weight_decay=weight_decay)):
	    if (network == 'ENet'):

                logits_val, probabilities_val = ENet(images_val,
                                                 num_classes,
                                                 batch_size=eval_batch_size,
                                                 is_training=True,
                                                 reuse=True,
                                                 num_initial_blocks=num_initial_blocks,
                                                 stage_two_repeat=stage_two_repeat,
                                                 skip_connections=skip_connections)
	    if (network == 'ENet_Small'):

                logits_val, probabilities_val = ENet_Small(images_val,
                                         num_classes,
                                         batch_size=eval_batch_size,
                                         is_training=True,
                                         reuse=True,
                                         num_initial_blocks=num_initial_blocks,
                                         skip_connections=skip_connections)

	    if (network == 'ErfNet'):

                logits_val, probabilities_val = ErfNet(images_val,
                                         num_classes,
                                         batch_size=eval_batch_size,
                                         is_training=True,
                                         reuse=True)

	    if (network == 'ErfNet_Small'):

                logits_val, probabilities_val = ErfNet_Small(images_val,
                                         num_classes,
                                         batch_size=eval_batch_size,
                                         is_training=True,
                                         reuse=True)

	    if (network == 'ErfNet_NoDS'):

                logits_val, probabilities_val = ErfNet_NoDS(images_val,
                                         num_classes,
                                         batch_size=eval_batch_size,
                                         is_training=True,
                                         reuse=True)

        #perform one-hot-encoding on the ground truth annotation to get same shape as the logits
        annotations_val = tf.cast(annotations_val, dtype=tf.int32)
        annotations_val = tf.reshape(annotations_val, shape=[eval_batch_size, image_height, image_width])
        annotations_ohe_val = tf.one_hot(annotations_val, num_classes, axis=-1)

        #State the metrics that you want to predict. We get a predictions that is not one_hot_encoded. ----> Should we use OHE instead?
        predictions_val = tf.argmax(probabilities_val, -1)
        accuracy_val, accuracy_val_update = tf.contrib.metrics.streaming_accuracy(predictions_val, annotations_val)
        mean_IOU_val, mean_IOU_val_update = tf.contrib.metrics.streaming_mean_iou(predictions=predictions_val, labels=annotations_val, num_classes=num_classes)

	weights_val = annotations_ohe_val * class_weights
	weights_val = tf.reduce_sum(weights_val, 3)
        mean_iIOU_val, mean_iIOU_val_update = tf.contrib.metrics.streaming_mean_iou(predictions=predictions_val, labels=annotations_val, num_classes=num_classes, weights = weights_val)
        metrics_op_val = tf.group(accuracy_val_update, mean_IOU_val_update, mean_iIOU_val_update)

        #Create an output for showing the segmentation output of validation images
        segmentation_output_val = tf.cast(predictions_val, dtype=tf.float32)
        segmentation_output_val = tf.reshape(segmentation_output_val, shape=[-1, image_height, image_width, 1])
        segmentation_ground_truth_val = tf.cast(annotations_val, dtype=tf.float32)
        segmentation_ground_truth_val = tf.reshape(segmentation_ground_truth_val, shape=[-1, image_height, image_width, 1])

        def eval_step(sess, metrics_op_val):
            '''
            Simply takes in a session, runs the metrics op and some logging information.
            '''
            start_time = time.time()
            accuracy_value, mean_IOU_value, mean_iIOU_value, _ = sess.run([accuracy_val, mean_IOU_val, mean_iIOU_val, metrics_op_val])
            time_elapsed = time.time() - start_time

            #Log some information
            logging.info('---VALIDATION--- Validation Accuracy: %.4f    Validation Mean IOU: %.4f    Validation Mean iIOU: %.4f	(%.2f sec/step)', accuracy_value, mean_IOU_value, mean_iIOU_value, time_elapsed)

            return accuracy_value, mean_IOU_value, mean_iIOU_value

        #=====================================================

        #Now finally create all the summaries you need to monitor and group them into one summary op.
        tf.summary.scalar('Monitor/Total_Loss', total_loss)
        tf.summary.scalar('Monitor/validation_accuracy', accuracy_val)
        tf.summary.scalar('Monitor/training_accuracy', accuracy)
        tf.summary.scalar('Monitor/validation_mean_IOU', mean_IOU_val)
        tf.summary.scalar('Monitor/training_mean_IOU', mean_IOU)
        tf.summary.scalar('Monitor/validation_mean_iIOU', mean_iIOU_val)
        tf.summary.scalar('Monitor/training_mean_iIOU', mean_iIOU)
        tf.summary.scalar('Monitor/learning_rate', lr)
        tf.summary.image('Images/Validation_original_image', images_val, max_outputs=1)
        tf.summary.image('Images/Validation_segmentation_output', segmentation_output_val, max_outputs=1)
        tf.summary.image('Images/Validation_segmentation_ground_truth', segmentation_ground_truth_val, max_outputs=1)
        my_summary_op = tf.summary.merge_all()

        #Define your supervisor for running a managed session. Do not run the summary_op automatically or else it will consume too much memory
        sv = tf.train.Supervisor(logdir=logdir, summary_op=None, saver=tf.train.Saver(max_to_keep=20),init_fn=None)

        # Run the managed session
        with sv.managed_session() as sess:
            for step in xrange(int(num_steps_per_epoch * num_epochs)):

                #At the start of every epoch, show the vital information:
                if step % num_batches_per_epoch == 0:
                    logging.info('Epoch %s/%s', step/num_batches_per_epoch + 1, num_epochs)
                    learning_rate_value = sess.run([lr])
                    logging.info('Current Learning Rate: %s', learning_rate_value)
            	
		#Check the validation data every val_step steps
		if ((step+1) % val_step == 0):
		    #for i in xrange(len(image_val_files) / eval_batch_size):
		    validation_accuracy, validation_mean_IOU, validation_mean_iIOU = eval_step(sess, metrics_op_val)

                #Log the summaries every log_step
                if ((step+1) % log_step == 0):
                    loss, training_accuracy, training_mean_IOU, training_mean_iIOU = train_step(sess, train_op, sv.global_step, metrics_op=metrics_op,print_step=print_step)

                    summaries = sess.run(my_summary_op)
                    sv.summary_computed(sess, summaries)
                    
                #If not, simply run the training step
                else:
                    loss, training_accuracy,training_mean_IOU, training_mean_iIOU = train_step(sess, train_op, sv.global_step, metrics_op=metrics_op,print_step=print_step)


		#Save checkpoint every checkpoint_step steps
            	if ((step+1) % checkpoint_step == 0):
		    logging.info('Saving model to disk...')
	            sv.saver.save(sess, sv.save_path, global_step = sv.global_step)

            #We log the final training loss
            logging.info('Final Loss: %s', loss)
            logging.info('Final Training Accuracy: %s', training_accuracy)
            logging.info('Final Training Mean IOU: %s', training_mean_IOU)
            logging.info('Final Training Mean iIOU: %s', training_mean_iIOU)
            logging.info('Final Validation Accuracy: %s', validation_accuracy)
            logging.info('Final Validation Mean IOU: %s', validation_mean_IOU)
            logging.info('Final Validation Mean iIOU: %s', validation_mean_iIOU)

            #Once all the training has been done, save the log files and checkpoint model
            logging.info('Finished training! Saving model to disk now.')
            sv.saver.save(sess, sv.save_path, global_step = sv.global_step)

            if save_images:
                if not os.path.exists(photo_dir):
                    os.mkdir(photo_dir)

                #Plot the predictions - check validation images only
                logging.info('Saving the images now...')
                predictions_value, annotations_value = sess.run([predictions_val, annotations_val])

                for i in xrange(eval_batch_size):
                    predicted_annotation = predictions_value[i]
                    annotation = annotations_value[i]

                    plt.subplot(1,2,1)
                    plt.imshow(predicted_annotation)
                    plt.subplot(1,2,2)
                    plt.imshow(annotation)
                    plt.savefig(photo_dir+"/image_" + str(i))

if __name__ == '__main__':
    run()
