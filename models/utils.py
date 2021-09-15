from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Cropping2D, Conv2DTranspose, BatchNormalization, Dropout, ReLU, Activation
from tensorflow.keras.layers import AveragePooling2D, Conv2D, Lambda
import keras.backend as K
import numpy as np


def pool_block(feats, pool_factor):

    h = K.int_shape(feats)[1]
    w = K.int_shape(feats)[2]

    pool_size = strides = [
        int(np.round(float(h) / pool_factor)),
        int(np.round(float(w) / pool_factor))]

    x = AveragePooling2D(pool_size, data_format='channels_last',
                         strides=strides, padding='same')(feats)
    x = Conv2D(512, (1, 1), data_format='channels_last',
               padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    outputs = Lambda(lambda y: K.resize_images(y,height_factor=strides[0],width_factor=strides[1],
    data_format='channels_last', interpolation='bilinear'))(x)
    return outputs


def crop(o1, o2, i):
    o_shape2 = Model(i, o2).output_shape

    output_height2 = o_shape2[1]
    output_width2 = o_shape2[2]

    o_shape1 = Model(i, o1).output_shape
    output_height1 = o_shape1[1]
    output_width1 = o_shape1[2]

    cx = abs(output_width1 - output_width2)
    cy = abs(output_height2 - output_height1)

    if output_width1 > output_width2:
        o1 = Cropping2D(cropping=((0, 0),  (0, cx)),
                        data_format='channels_last')(o1)
    else:
        o2 = Cropping2D(cropping=((0, 0),  (0, cx)),
                        data_format='channels_last')(o2)

    if output_height1 > output_height2:
        o1 = Cropping2D(cropping=((0, cy),  (0, 0)),
                        data_format='channels_last')(o1)
    else:
        o2 = Cropping2D(cropping=((0, cy),  (0, 0)),
                        data_format='channels_last')(o2)

    return o1, o2

def upsample(filters, size, norm_type='batchnorm', apply_dropout=False):
  """Upsamples an input.
  Conv2DTranspose => Batchnorm => Dropout => Relu
  Args:
    filters: number of filters
    size: filter size
    norm_type: Normalization type; either 'batchnorm' or 'instancenorm'.
    apply_dropout: If True, adds the dropout layer
  Returns:
    Upsample Sequential Model
  """

  initializer = tf.random_normal_initializer(0., 0.02)

  result = Sequential()
  result.add(
      Conv2DTranspose(filters, size, strides=2,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      use_bias=False))

  if norm_type.lower() == 'batchnorm':
    result.add(BatchNormalization())

  if apply_dropout:
    result.add(Dropout(0.5))

  result.add(ReLU())

  return result