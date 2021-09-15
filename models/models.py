import os
import numpy as np
import keras.backend as K
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model, Input, Sequential
from tensorflow.keras.layers import Conv2DTranspose, BatchNormalization, Dropout, ReLU, Concatenate, Lambda, ZeroPadding2D
from tensorflow.keras.utils import get_file
from tensorflow.keras.applications import MobileNetV2
import tensorflow_hub as hub


from keras.regularizers import l2
from keras.layers import *
from keras.engine import Layer
from keras.applications.vgg16 import *
from keras.models import *
from .mobilenet import get_mobilenet_encoder
from .utils import upsample, crop, pool_block

#data_augmentation = Sequential(
#    [
#        layers.RandomFlip(),
#        layers.RandomRotation(0.2),
#        layers.RandomTranslation(0.2,0.2),
#    ],
#    name="data_augmentation",
#)

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

    
    