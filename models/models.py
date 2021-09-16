import os
import numpy as np
import keras.backend as K
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model, Input, Sequential
from tensorflow.keras.layers import Conv2DTranspose, BatchNormalization, Dropout, ReLU, Concatenate, Lambda, ZeroPadding2D
from tensorflow.keras.utils import get_file
from tensorflow.keras.applications import MobileNetV2
from tensorflow.python.keras.utils.layer_utils import get_source_inputs
import tensorflow_hub as hub


from keras.regularizers import l2
from keras.layers import *
from keras.engine import Layer
from keras.applications.vgg16 import *
from keras.models import *
from .mobilenet import get_mobilenet_encoder
from .utils import _make_divisible, _inverted_res_block
from .utils import pool_block, crop, upsample

#data_augmentation = Sequential(
#    [
#        layers.RandomFlip(),
#        layers.RandomRotation(0.2),
#        layers.RandomTranslation(0.2,0.2),
#    ],
#    name="data_augmentation",
#)

WEIGHTS_PATH_MOBILE = "https://github.com/bonlime/keras-deeplab-v3-plus/releases/download/1.1/deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels.h5"

def Deeplabv3(input_shape, output_channels):
    """ Instantiates the Deeplabv3+ architecture
    Optionally loads weights pre-trained
    on PASCAL VOC. This model is available for TensorFlow only.
    # Arguments
        input_shape: shape of input image. format HxWxC
            PASCAL VOC model was trained on (512,512,3) images. None is allowed as shape/width
        output_channels: number of desired classes. PASCAL VOC has 21 classes.
            If number of classes not aligned with the weights used, last layer is initialized randomly
    # Returns
        A Keras model instance.
    # Raises
        RuntimeError: If attempting to run this model with a
            backend that does not support separable convolutions.
        ValueError: in case of invalid argument for `weights` or `backbone`
    """

    alpha=1.
    activation=None
    weights='pascal_voc'
    #backbone='mobilenetv2'

    img_input = Input(shape=input_shape)

    first_block_filters = _make_divisible(32 * alpha, 8)
    x = Conv2D(first_block_filters,
                kernel_size=3,
                strides=(2, 2), padding='same', use_bias=False,
                name='Conv' if input_shape[2] == 3 else 'Conv_')(img_input)
    x = BatchNormalization(
        epsilon=1e-3, momentum=0.999, name='Conv_BN')(x)
    x = Activation(tf.nn.relu6, name='Conv_Relu6')(x)

    x = _inverted_res_block(x, filters=16, alpha=alpha, stride=1,
                            expansion=1, block_id=0, skip_connection=False)

    x = _inverted_res_block(x, filters=24, alpha=alpha, stride=2,
                            expansion=6, block_id=1, skip_connection=False)
    x = _inverted_res_block(x, filters=24, alpha=alpha, stride=1,
                            expansion=6, block_id=2, skip_connection=True)

    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=2,
                            expansion=6, block_id=3, skip_connection=False)
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                            expansion=6, block_id=4, skip_connection=True)
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                            expansion=6, block_id=5, skip_connection=True)

    # stride in block 6 changed from 2 -> 1, so we need to use rate = 2
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,  # 1!
                            expansion=6, block_id=6, skip_connection=False)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, rate=2,
                            expansion=6, block_id=7, skip_connection=True)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, rate=2,
                            expansion=6, block_id=8, skip_connection=True)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, rate=2,
                            expansion=6, block_id=9, skip_connection=True)

    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=2,
                            expansion=6, block_id=10, skip_connection=False)
    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=2,
                            expansion=6, block_id=11, skip_connection=True)
    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=2,
                            expansion=6, block_id=12, skip_connection=True)

    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=2,  # 1!
                            expansion=6, block_id=13, skip_connection=False)
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=4,
                            expansion=6, block_id=14, skip_connection=True)
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=4,
                            expansion=6, block_id=15, skip_connection=True)

    x = _inverted_res_block(x, filters=320, alpha=alpha, stride=1, rate=4,
                            expansion=6, block_id=16, skip_connection=False)

    # end of feature extractor

    # branching for Atrous Spatial Pyramid Pooling

    # Image Feature branch
    shape_before = tf.shape(x)
    b4 = GlobalAveragePooling2D()(x)
    b4_shape = tf.keras.backend.int_shape(b4)
    # from (b_size, channels)->(b_size, 1, 1, channels)
    b4 = Reshape((1, 1, b4_shape[1]))(b4)
    b4 = Conv2D(256, (1, 1), padding='same',
                use_bias=False, name='image_pooling')(b4)
    b4 = BatchNormalization(name='image_pooling_BN', epsilon=1e-5)(b4)
    b4 = Activation(tf.nn.relu)(b4)
    # upsample. have to use compat because of the option align_corners
    size_before = tf.keras.backend.int_shape(x)
    b4 = tf.keras.layers.experimental.preprocessing.Resizing(
            *size_before[1:3], interpolation="bilinear"
        )(b4)
    # simple 1x1
    b0 = Conv2D(256, (1, 1), padding='same', use_bias=False, name='aspp0')(x)
    b0 = BatchNormalization(name='aspp0_BN', epsilon=1e-5)(b0)
    b0 = Activation(tf.nn.relu, name='aspp0_activation')(b0)

    # there are only 2 branches in mobilenetV2. not sure why
    x = Concatenate()([b4, b0])

    x = Conv2D(256, (1, 1), padding='same',
               use_bias=False, name='concat_projection')(x)
    x = BatchNormalization(name='concat_projection_BN', epsilon=1e-5)(x)
    x = Activation(tf.nn.relu)(x)
    x = Dropout(0.1)(x)
    # DeepLab v.3+ decoder

    # you can use it with arbitary number of output_channels
    if (weights == 'pascal_voc' and output_channels == 21):
        last_layer_name = 'logits_semantic'
    else:
        last_layer_name = 'custom_logits_semantic'

    x = Conv2D(output_channels, (1, 1), padding='same', name=last_layer_name)(x)
    size_before3 = tf.keras.backend.int_shape(img_input)
    x = tf.keras.layers.experimental.preprocessing.Resizing(
            *size_before3[1:3], interpolation="bilinear"
        )(x)


    inputs = img_input

    if activation in {'softmax', 'sigmoid'}:
        x = tf.keras.layers.Activation(activation)(x)

    model = Model(inputs, x, name='Deeplabv3')

    # load weights

    if weights == 'pascal_voc':
        weights_path = get_file('deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels.h5',
                                    WEIGHTS_PATH_MOBILE,
                                    cache_subdir='models')
        model.load_weights(weights_path, by_name=True)
    return model


def mobilenet_segnet(input_shape, output_channels):

    input_height=input_shape[0]
    input_width=input_shape[1]
    channels=input_shape[2]

    # encoder
    img_input, levels = get_mobilenet_encoder(
        input_height=input_height,  input_width=input_width, channels=channels)

    # decoder
    o = levels[3]
    o = (ZeroPadding2D((1, 1), data_format='channels_last'))(o)
    o = (Conv2D(512, (3, 3), padding='valid', data_format='channels_last'))(o)
    o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2), data_format='channels_last'))(o)
    o = (ZeroPadding2D((1, 1), data_format='channels_last'))(o)
    o = (Conv2D(256, (3, 3), padding='valid', data_format='channels_last'))(o)
    o = (BatchNormalization())(o)

    for _ in range(2):
        o = (UpSampling2D((2, 2), data_format='channels_last'))(o)
        o = (ZeroPadding2D((1, 1), data_format='channels_last'))(o)
        o = (Conv2D(128, (3, 3), padding='valid',
              data_format='channels_last'))(o)
        o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2), data_format='channels_last'))(o)
    o = (ZeroPadding2D((1, 1), data_format='channels_last'))(o)
    o = (Conv2D(64, (3, 3), padding='valid', data_format='channels_last', name="seg_feats"))(o)
    o = (BatchNormalization())(o)

    outputs = Conv2D(output_channels, (3, 3), padding='same',
                data_format='channels_last')(o)

    model = Model(inputs=img_input, outputs=outputs, name="mobilenet_segnet")
    return model

def mobileNetV2_segnet(input_shape, output_channels):

    input_height=input_shape[0]
    input_width=input_shape[1]
    channels=input_shape[2]

    #Build the encoder
    base_model = MobileNetV2(
                    include_top=False,
                    weights="imagenet",  # Load weights pre-trained on ImageNet.
                    input_shape=input_shape,
    )  # Do not include the ImageNet classifier at the top.

    # Use the activations of these layers
    layer_names = [
      'block_1_expand_relu',   # 64x64
      'block_3_expand_relu',   # 32x32
      'block_6_expand_relu',   # 16x16
      'block_13_expand_relu',  # 8x8
      'block_16_project',      # 4x4
    ] 

    base_models_output = [base_model.get_layer(name).output for name in layer_names]
    encoder = Model(base_model.input, base_models_output, name="encoder")

    # Freeze the base_model
    encoder.trainable = False

    #Build the model
    img_input = Input(shape=input_shape)
    #x = data_augmentation(img_input)
    x = img_input
    # Downsampling through the model
    levels = encoder(x, training=False)

    # decoder
    o = levels[3]
    o = (ZeroPadding2D((1, 1), data_format='channels_last'))(o)
    o = (Conv2D(512, (3, 3), padding='valid', data_format='channels_last'))(o)
    o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2), data_format='channels_last'))(o)
    o = (ZeroPadding2D((1, 1), data_format='channels_last'))(o)
    o = (Conv2D(256, (3, 3), padding='valid', data_format='channels_last'))(o)
    o = (BatchNormalization())(o)

    for _ in range(2):
        o = (UpSampling2D((2, 2), data_format='channels_last'))(o)
        o = (ZeroPadding2D((1, 1), data_format='channels_last'))(o)
        o = (Conv2D(128, (3, 3), padding='valid',
              data_format='channels_last'))(o)
        o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2), data_format='channels_last'))(o)
    o = (ZeroPadding2D((1, 1), data_format='channels_last'))(o)
    o = (Conv2D(64, (3, 3), padding='valid', data_format='channels_last', name="seg_feats"))(o)
    o = (BatchNormalization())(o)

    outputs = Conv2D(output_channels, (3, 3), padding='same',
                data_format='channels_last')(o)

    model = Model(inputs=img_input, outputs=outputs, name="mobileNetV2_segnet")
    return model

def mobilenet_pspnet(input_shape, output_channels):

    input_height=input_shape[0]
    input_width=input_shape[1]
    channels=input_shape[2]

    #assert input_height % 192 == 0
    #assert input_width % 192 == 0

    img_input, levels = get_mobilenet_encoder(
        input_height=input_height,  input_width=input_width, channels=channels)
        
    [f1, f2, f3, f4, f5] = levels

    o = f5

    pool_factors = [1, 2, 3, 6]
    pool_outs = [o]

    for p in pool_factors:
        pooled = pool_block(o, p)
        pool_outs.append(pooled)

    o = Concatenate(axis=-1)(pool_outs)

    o = Conv2D(512, (1, 1), data_format='channels_last', use_bias=False , name="seg_feats" )(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)

    o = Conv2D(output_channels, (3, 3), data_format='channels_last',
               padding='same')(o)

    outputs = Lambda(lambda x: K.resize_images(x,height_factor=32,width_factor=32,
    data_format='channels_last', interpolation='bilinear'))(o)

    model = Model(inputs=inputs, outputs=outputs, name="mobilenet_pspnet")
    return model

def mobileNetV2_pspnet(input_shape, output_channels):

    input_height=input_shape[0]
    input_width=input_shape[1]
    channels=input_shape[2]

    #Build the encoder
    base_model = MobileNetV2(
                    include_top=False,
                    weights="imagenet",  # Load weights pre-trained on ImageNet.
                    input_shape=input_shape,
    )  # Do not include the ImageNet classifier at the top.

    # Use the activations of these layers
    layer_names = [
      'block_1_expand_relu',   # 64x64
      'block_3_expand_relu',   # 32x32
      'block_6_expand_relu',   # 16x16
      'block_13_expand_relu',  # 8x8
      'block_16_project',      # 4x4
    ] 

    base_models_output = [base_model.get_layer(name).output for name in layer_names]
    encoder = Model(base_model.input, base_models_output, name="encoder")

    # Freeze the base_model
    encoder.trainable = False

    #Build the model
    img_input = Input(shape=input_shape)
    #x = data_augmentation(img_input)
    x = img_input
    # Downsampling through the model
    levels = encoder(x, training=False)
        
    [f1, f2, f3, f4, f5] = levels

    o = f5

    pool_factors = [1, 2, 3, 6]
    pool_outs = [o]

    for p in pool_factors:
        pooled = pool_block(o, p)
        pool_outs.append(pooled)

    o = Concatenate(axis=-1)(pool_outs)

    o = Conv2D(512, (1, 1), data_format='channels_last', use_bias=False , name="seg_feats" )(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)

    o = Conv2D(output_channels, (3, 3), data_format='channels_last',
               padding='same')(o)

    outputs = Lambda(lambda x: K.resize_images(x,height_factor=32,width_factor=32,
    data_format='channels_last', interpolation='bilinear'))(o)

    model = Model(inputs=img_input, outputs=outputs, name="mobileNetV2_pspnet")
    return model


def mobilenet_fcn_8(input_shape, output_channels):

    input_height=input_shape[0]
    input_width=input_shape[1]
    channels=input_shape[2]

    img_input, levels = get_mobilenet_encoder(
        input_height=input_height,  input_width=input_width, channels=channels)
    [f1, f2, f3, f4, f5] = levels

    o = f5

    o = (Conv2D(4096, (7, 7), activation='relu',
                padding='same', data_format='channels_last'))(o)
    o = Dropout(0.5)(o)
    o = (Conv2D(4096, (1, 1), activation='relu',
                padding='same', data_format='channels_last'))(o)
    o = Dropout(0.5)(o)

    o = (Conv2D(output_channels,  (1, 1), kernel_initializer='he_normal',
                data_format='channels_last'))(o)
    o = Conv2DTranspose(output_channels, kernel_size=(4, 4),  strides=(
        2, 2), use_bias=False, data_format='channels_last')(o)

    o2 = f4
    o2 = (Conv2D(output_channels,  (1, 1), kernel_initializer='he_normal',
                 data_format='channels_last'))(o2)

    o, o2 = crop(o, o2, img_input)

    o = Add()([o, o2])

    o = Conv2DTranspose(output_channels, kernel_size=(4, 4),  strides=(
        2, 2), use_bias=False, data_format='channels_last')(o)
    o2 = f3
    o2 = (Conv2D(output_channels,  (1, 1), kernel_initializer='he_normal',
                 data_format='channels_last'))(o2)
    o2, o = crop(o2, o, img_input)
    o = Add( name="seg_feats" )([o2, o])

    outputs = Conv2DTranspose(output_channels, kernel_size=(8, 8),  strides=(
        8, 8), use_bias=False, data_format='channels_last')(o)

    model = Model(inputs=img_input, outputs=outputs, name="mobilenet_fcn_8")
    return model



def mobilenet_fcn_32(input_shape, output_channels):

    input_height=input_shape[0]
    input_width=input_shape[1]
    channels=input_shape[2]

    img_input, levels = get_mobilenet_encoder(
        input_height=input_height,  input_width=input_width, channels=channels)
    [f1, f2, f3, f4, f5] = levels

    o = f5

    o = (Conv2D(4096, (7, 7), activation='relu',
                padding='same', data_format='channels_last'))(o)
    o = Dropout(0.5)(o)
    o = (Conv2D(4096, (1, 1), activation='relu',
                padding='same', data_format='channels_last'))(o)
    o = Dropout(0.5)(o)

    o = (Conv2D(output_channels,  (1, 1), kernel_initializer='he_normal',
                data_format='channels_last' , name="seg_feats" ))(o)
    outputs = Conv2DTranspose(output_channels, kernel_size=(32, 32),  strides=(
        32, 32), use_bias=False,  data_format='channels_last')(o)

    model = Model(inputs=img_input, outputs=outputs, name="mobilenet_fcn_32")
    return model

def mobileNetV2_Unet(input_shape, output_channels):
    
    #Build the encoder
    base_model = MobileNetV2(
                    include_top=False,
                    weights="imagenet",  # Load weights pre-trained on ImageNet.
                    input_shape=input_shape,
    )  # Do not include the ImageNet classifier at the top.

    # Use the activations of these layers
    layer_names = [
      'block_1_expand_relu',   # 64x64
      'block_3_expand_relu',   # 32x32
      'block_6_expand_relu',   # 16x16
      'block_13_expand_relu',  # 8x8
      'block_16_project',      # 4x4
    ] 

    base_models_output = [base_model.get_layer(name).output for name in layer_names]
    encoder = Model(base_model.input, base_models_output, name="encoder")

    # Freeze the base_model
    encoder.trainable = False

    #Build the decoder stack
    decoder_stack = [
      upsample(512, 3),  # 4x4 -> 8x8
      upsample(256, 3),  # 8x8 -> 16x16
      upsample(128, 3),  # 16x16 -> 32x32
      upsample(64, 3),   # 32x32 -> 64x64
    ]
    #Build the model
    inputs = Input(shape=input_shape)
    #x = data_augmentation(inputs)
    x = inputs
    # Downsampling through the model
    skips = encoder(x, training=False)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(decoder_stack, skips):
      x = up(x)
      concat = Concatenate()
      x = concat([x, skip])

    # This is the last layer of the model
    last = Conv2DTranspose(
        filters=output_channels, kernel_size=3, strides=2,
        padding='same')  #64x64 -> 128x128

    outputs = last(x)
    
    model = Model(inputs=inputs, outputs=outputs, name="mobileNetV2_Unet")
    return model

if __name__ == '__main__':
    mobileNetV2((32, 32, 3), 3).summary()

    
    