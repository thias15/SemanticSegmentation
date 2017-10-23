import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers
slim = tf.contrib.slim

'''
=============================================================
ErfNet: Efficient ConvNet for Real-time Semantic Segmentation
=============================================================
Based on the paper: http://www.robesafe.uah.es/personal/eduardo.romera/pdfs/Romera17iv.pdf
'''

def spatial_dropout(x, p, seed, scope, is_training=True):
    '''
    Performs a 2D spatial dropout that drops layers instead of individual elements in an input feature map.
    Note that p stands for the probability of dropping, but tf.nn.relu uses probability of keeping.

    ------------------
    Technical Details
    ------------------
    The noise shape must be of shape [batch_size, 1, 1, num_channels], with the height and width set to 1, because
    it will represent either a 1 or 0 for each layer, and these 1 or 0 integers will be broadcasted to the entire
    dimensions of each layer they interact with such that they can decide whether each layer should be entirely
    'dropped'/set to zero or have its activations entirely kept.
    --------------------------

    INPUTS:
    - x(Tensor): a 4D Tensor of the input feature map.
    - p(float): a float representing the probability of dropping a layer
    - seed(int): an integer for random seeding the random_uniform distribution that runs under tf.nn.relu
    - scope(str): the string name for naming the spatial_dropout
    - is_training(bool): to turn on dropout only when training. Optional.

    OUTPUTS:
    - output(Tensor): a 4D Tensor that is in exactly the same size as the input x,
                      with certain layers having their elements all set to 0 (i.e. dropped).
    '''
    if is_training:
        keep_prob = 1.0 - p
        input_shape = x.get_shape().as_list()
        noise_shape = tf.constant(value=[input_shape[0], 1, 1, input_shape[3]])
        output = tf.nn.dropout(x, keep_prob, noise_shape, seed=seed, name=scope)

        return output

    return x

@slim.add_arg_scope
def downsampler(inputs, output_depth, is_training=True, scope='downsampler'):
    '''
    INPUTS:
    - inputs(Tensor): A 4D tensor of shape [batch_size, height, width, channels]
    - output_depth(int): an integer indicating the output depth of the output convolutional block.

    OUTPUTS:
    - net_concatenated(Tensor): a 4D Tensor that contains the 
    '''
    input_shape = inputs.get_shape().as_list()
    input_depth = input_shape[-1]

    #Convolutional branch
    net_conv = slim.conv2d(inputs, output_depth-input_depth, [3,3], stride=2, activation_fn=None, scope=scope+'_conv')

    #Max pool branch
    net_pool = slim.max_pool2d(inputs, [2,2], stride=2, scope=scope+'_max_pool')

    #Concatenated output
    net = tf.concat([net_conv, net_pool], axis=3, name=scope+'_concat')

    net = slim.batch_norm(net_conv, is_training=is_training, fused=None, scope=scope+'_batch_norm')
    net = tf.nn.relu(net, name=scope+'_relu')

    return net

@slim.add_arg_scope
def upsampler(inputs, output_depth, is_training=True, scope='downsampler'):
    '''
    INPUTS:
    - inputs(Tensor): A 4D tensor of shape [batch_size, height, width, channels]
    - output_depth(int): an integer indicating the output depth of the output convolutional block.

    OUTPUTS:
    - net_concatenated(Tensor): a 4D Tensor that contains the 
    '''
    input_shape = inputs.get_shape().as_list()
    input_depth = input_shape[-1]

    #Convolutional branch
    net_conv = slim.conv2d(inputs, output_depth-input_depth, [3,3], stride=2, activation_fn=None, scope=scope+'_conv')

    #Max pool branch
    net_pool = slim.max_pool2d(inputs, [2,2], stride=2, scope=scope+'_max_pool')

    #Concatenated output
    net = tf.concat([net_conv, net_pool], axis=3, name=scope+'_concat')

    net = slim.batch_norm(net_conv, is_training=is_training, fused=None, scope=scope+'_batch_norm')
    net = tf.nn.relu(net, name=scope+'_relu')

    return net

@slim.add_arg_scope
def non_bottleneck(inputs,
               output_depth,
               filter_size=3,
               regularizer_prob=0.3,
               seed=0,
               is_training=True,
               dilation_rate=None,
               scope='non_bottleneck'):
    '''
    INPUTS:
    - inputs(Tensor): a 4D Tensor of the previous convolutional block of shape [batch_size, height, width, num_channels].
    - output_depth(int): an integer indicating the output depth of the output convolutional block.
    - filter_size(int): an integer that gives the height and width of the filter size to use for a regular/dilated convolution.
    - regularizer_prob(float): the float p that represents the prob of dropping a layer for spatial dropout regularization.
    - seed(int): an integer for the random seed used in the random normal distribution within dropout.
    - is_training(bool): a boolean value to indicate whether or not is training. Decides batch_norm and prelu activity.
    - dilation_rate(int): the dilation factor for performing atrous convolution/dilated convolution.
    - scope(str): a string name that names your bottleneck.

    OUTPUTS:
    - net(Tensor): The convolution block output after a bottleneck
    - pooling_indices(Tensor): If downsample, then this tensor is produced for use in upooling later.
    - inputs_shape(list): The shape of the input to the downsampling conv block. For use in unpooling later.

    '''

    #============NON-BOTTLENECK 1D====================
    #Check if dilation rate is given
    if not dilation_rate:
        raise ValueError('Dilation rate is not given.')

    #Save the main branch for addition later
    net_main = inputs

    #First conv block - asymmetric convolution
    net = slim.conv2d(inputs, output_depth, [filter_size, 1], scope=scope+'_conv1a')
    net = tf.nn.relu(net, name=scope+'_relu1a')
    net = slim.conv2d(inputs, output_depth, [1,filter_size], scope=scope+'_conv1b')
    net = slim.batch_norm(net, is_training=is_training, fused=None, scope=scope+'_batch_norm1')
    net = tf.nn.relu(net, name=scope+'_relu1b')

    #Second conv block - asymmetric + dilation convolution
    net = slim.conv2d(inputs, output_depth, [filter_size, 1], rate=[dilation_rate,1], scope=scope+'_conv2a')
    net = tf.nn.relu(net, name=scope+'_relu2a')
    net = slim.conv2d(inputs, output_depth, [1,filter_size], rate=[1,dilation_rate], scope=scope+'_conv2b')
    net = slim.batch_norm(net, is_training=is_training, fused=None, scope=scope+'_batch_norm2')

    #Regularizer
    net = spatial_dropout(net, p=regularizer_prob, seed=seed, scope=scope+'_spatial_dropout')

    #Add the main branch
    net = tf.add(net_main, net, name=scope+'_add_residual')
    net = tf.nn.relu(net, name=scope+'_relu2b')

    return net

 
#Now actually start building the network
def ErfNet(inputs,
         num_classes,
         batch_size,
         num_initial_blocks=1,
         stage_two_repeat=2,
         skip_connections=True,
         reuse=None,
         is_training=True,
         scope='ENet'):
    '''
    The ENet model for real-time semantic segmentation!

    INPUTS:
    - inputs(Tensor): a 4D Tensor of shape [batch_size, image_height, image_width, num_channels] that represents one batch of preprocessed images.
    - num_classes(int): an integer for the number of classes to predict. This will determine the final output channels as the answer.
    - batch_size(int): the batch size to explictly set the shape of the inputs in order for operations to work properly.
    - num_initial_blocks(int): the number of times to repeat the initial block.
    - stage_two_repeat(int): the number of times to repeat stage two in order to make the network deeper.
    - skip_connections(bool): if True, add the corresponding encoder feature maps to the decoder. They are of exact same shapes.
    - reuse(bool): Whether or not to reuse the variables for evaluation.
    - is_training(bool): if True, switch on batch_norm and prelu only during training, otherwise they are turned off.
    - scope(str): a string that represents the scope name for the variables.

    OUTPUTS:
    - net(Tensor): a 4D Tensor output of shape [batch_size, image_height, image_width, num_classes], where each pixel has a one-hot encoded vector
                      determining the label of the pixel.
    '''
    #Set the shape of the inputs first to get the batch_size information
    inputs_shape = inputs.get_shape().as_list()
    inputs.set_shape(shape=(batch_size, inputs_shape[1], inputs_shape[2], inputs_shape[3]))

    with tf.variable_scope(scope, reuse=reuse):
        #Set the primary arg scopes. Fused batch_norm is faster than normal batch norm.
        with slim.arg_scope([initial_block, bottleneck], is_training=is_training),\
             slim.arg_scope([slim.batch_norm], fused=True), \
             slim.arg_scope([slim.conv2d, slim.conv2d_transpose], activation_fn=None): 
            #=================INITIAL BLOCK=================
            for i in xrange(1, max(num_initial_blocks, 1) + 1):
                net = initial_block(inputs, scope='initial_block_' + str(i))

            #Save for skip connection later
            if skip_connections:
                net_one = net

            #===================STAGE ONE=======================
            net, pooling_indices_1, inputs_shape_1 = bottleneck(net, output_depth=64, filter_size=3, regularizer_prob=0.01, downsampling=True, scope='bottleneck1_0')
            net = bottleneck(net, output_depth=64, filter_size=3, regularizer_prob=0.01, scope='bottleneck1_1')
            net = bottleneck(net, output_depth=64, filter_size=3, regularizer_prob=0.01, scope='bottleneck1_2')
            net = bottleneck(net, output_depth=64, filter_size=3, regularizer_prob=0.01, scope='bottleneck1_3')
            net = bottleneck(net, output_depth=64, filter_size=3, regularizer_prob=0.01, scope='bottleneck1_4')

            #Save for skip connection later
            if skip_connections:
                net_two = net

            #regularization prob is 0.1 from bottleneck 2.0 onwards
            with slim.arg_scope([bottleneck], regularizer_prob=0.1):
                net, pooling_indices_2, inputs_shape_2 = bottleneck(net, output_depth=128, filter_size=3, downsampling=True, scope='bottleneck2_0')
                
                #Repeat the stage two at least twice to get stage 2 and 3:
                for i in xrange(2, max(stage_two_repeat, 2) + 2):
                    net = bottleneck(net, output_depth=128, filter_size=3, scope='bottleneck'+str(i)+'_1')
                    net = bottleneck(net, output_depth=128, filter_size=3, dilated=True, dilation_rate=2, scope='bottleneck'+str(i)+'_2')
                    net = bottleneck(net, output_depth=128, filter_size=5, asymmetric=True, scope='bottleneck'+str(i)+'_3')
                    net = bottleneck(net, output_depth=128, filter_size=3, dilated=True, dilation_rate=4, scope='bottleneck'+str(i)+'_4')
                    net = bottleneck(net, output_depth=128, filter_size=3, scope='bottleneck'+str(i)+'_5')
                    net = bottleneck(net, output_depth=128, filter_size=3, dilated=True, dilation_rate=8, scope='bottleneck'+str(i)+'_6')
                    net = bottleneck(net, output_depth=128, filter_size=5, asymmetric=True, scope='bottleneck'+str(i)+'_7')
                    net = bottleneck(net, output_depth=128, filter_size=3, dilated=True, dilation_rate=16, scope='bottleneck'+str(i)+'_8')

            with slim.arg_scope([bottleneck], regularizer_prob=0.1, decoder=True):
                #===================STAGE FOUR========================
                bottleneck_scope_name = "bottleneck" + str(i + 1)

                #The decoder section, so start to upsample.
                net = bottleneck(net, output_depth=64, filter_size=3, upsampling=True,
                                 pooling_indices=pooling_indices_2, output_shape=inputs_shape_2, scope=bottleneck_scope_name+'_0')

                #Perform skip connections here
                if skip_connections:
                    net = tf.add(net, net_two, name=bottleneck_scope_name+'_skip_connection')

                net = bottleneck(net, output_depth=64, filter_size=3, scope=bottleneck_scope_name+'_1')
                net = bottleneck(net, output_depth=64, filter_size=3, scope=bottleneck_scope_name+'_2')

                #===================STAGE FIVE========================
                bottleneck_scope_name = "bottleneck" + str(i + 2)

                net = bottleneck(net, output_depth=16, filter_size=3, upsampling=True,
                                 pooling_indices=pooling_indices_1, output_shape=inputs_shape_1, scope=bottleneck_scope_name+'_0')

                #perform skip connections here
                if skip_connections:
                    net = tf.add(net, net_one, name=bottleneck_scope_name+'_skip_connection')

                net = bottleneck(net, output_depth=16, filter_size=3, scope=bottleneck_scope_name+'_1')

            #=============FINAL CONVOLUTION=============
            logits = slim.conv2d_transpose(net, num_classes, [2,2], stride=2, scope='fullconv')
            probabilities = tf.nn.softmax(logits, name='logits_to_softmax')

        return logits, probabilities


def ENet_arg_scope(weight_decay=2e-4,
                   batch_norm_decay=0.1,
                   batch_norm_epsilon=0.001):
  '''
  The arg scope for enet model. The weight decay is 2e-4 as seen in the paper.
  Batch_norm decay is 0.1 (momentum 0.1) according to official implementation.

  INPUTS:
  - weight_decay(float): the weight decay for weights variables in conv2d and separable conv2d
  - batch_norm_decay(float): decay for the moving average of batch_norm momentums.
  - batch_norm_epsilon(float): small float added to variance to avoid dividing by zero.

  OUTPUTS:
  - scope(arg_scope): a tf-slim arg_scope with the parameters needed for xception.
  '''
  # Set weight_decay for weights in conv2d and separable_conv2d layers.
  with slim.arg_scope([slim.conv2d],
                      weights_regularizer=slim.l2_regularizer(weight_decay),
                      biases_regularizer=slim.l2_regularizer(weight_decay)):

    # Set parameters for batch_norm.
    with slim.arg_scope([slim.batch_norm],
                        decay=batch_norm_decay,
                        epsilon=batch_norm_epsilon) as scope:
      return scope
