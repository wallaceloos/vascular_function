import numpy as np

import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
from tensorflow.keras.layers import Conv3D, MaxPool3D, concatenate, Dropout, Lambda, UpSampling3D

def loss_mae(y_true, y_pred):

    flatten = tf.keras.layers.Flatten()
    y_true_f = flatten(y_true)
    y_pred_f = flatten(y_pred)

    loss = tf.keras.losses.MAE(y_true_f, y_pred_f)

    return 200*loss

def loss_mae_without_factor(y_true, y_pred):#without factor ajustment
    flatten = tf.keras.layers.Flatten()
    y_true_f = flatten(y_true)
    y_pred_f = flatten(y_pred)

    loss = tf.keras.losses.MAE(y_true_f, y_pred_f)
    return loss

def castTensor(tensor):

    tensor = tf.cast(tensor, tf.float32)

    return tensor

def ROIs(tensor):

    tensor1 = tensor[0]
    tensor2 = tensor[1]
    ans = tf.math.multiply(tensor1,tensor2)
    ans = tf.cast(ans, tf.float32)

    return ans


def computeCurve(tensor):

    mask = tensor[0]
    roi = tensor[1]

    num = tf.keras.backend.sum(roi, axis=(1, 2, 3), keepdims=False)
    den = tf.keras.backend.sum(mask, axis=(1, 2, 3), keepdims=False)
    curve = tf.math.divide(num,den+1e-8)
    curve = tf.cast(curve, tf.float32)

    return curve

def normalizeOutput(tensor):

    tensor_norm = (tensor-tf.reduce_min(tensor))/( tf.reduce_max(tensor) - tf.reduce_min(tensor) + 1e-10)
    return tensor_norm

def loss_computeCofDistance3D(y_true, y_pred):

    cof = y_true
    mask = y_pred
    cof = tf.cast(cof, tf.float32)
    mask = tf.cast(mask, tf.float32)

    ii, jj, zz, _ = tf.meshgrid(tf.range(120), tf.range(120), tf.range(120), tf.range(1), indexing='ij')
    ii = tf.cast(ii, tf.float32)
    jj = tf.cast(jj, tf.float32)
    zz = tf.cast(zz, tf.float32)

    dx = (ii-cof[:,0])**2
    dy = (jj-cof[:,1])**2
    dz = (zz-cof[:,2])**2

    dtotal = (dx+dy+dz)
    dtotal = tf.math.sqrt(dtotal)
    dtotal = tf.math.multiply(dtotal,mask)
    dtotal = tf.reduce_sum(dtotal, axis=(1,2,3,4))

    return dtotal/(tf.reduce_sum(mask)+1e-10)#this division is made to avoid a trivial solution (mask all zeros)

def unet3d(img_size = (None, None, None),learning_rate = 1e-8,\
                 learning_decay = 1e-8, drop_out = 0.35,nchannels = 7):

    dropout = drop_out
    input_img = tf.keras.layers.Input((img_size[0], img_size[1], img_size[2], nchannels))

    #encoder
    conv1_1 = Conv3D(32, (11, 11, 11), activation=keras.layers.LeakyReLU(alpha=0.3), padding='same')(input_img)
    conv1_2 = Conv3D(32, (11, 11, 11), activation=keras.layers.LeakyReLU(alpha=0.3), padding='same')(conv1_1)
    conv1_2 = tfa.layers.InstanceNormalization()(conv1_2)
    pool1 = MaxPool3D(pool_size=(2, 2, 2))(conv1_2)

    conv2_1 = Conv3D(64, (7, 7, 7), activation=keras.layers.LeakyReLU(alpha=0.3), padding='same')(pool1)
    conv2_2 = Conv3D(64, (7, 7, 7), activation=keras.layers.LeakyReLU(alpha=0.3), padding='same')(conv2_1)
    pool2 = MaxPool3D(pool_size=(2, 2, 2))(conv2_2)

    conv3_1 = Conv3D(128, (7, 7, 7), activation=keras.layers.LeakyReLU(alpha=0.3), padding='same')(pool2)
    conv3_2 = Conv3D(128, (7, 7, 7), activation=keras.layers.LeakyReLU(alpha=0.3), padding='same')(conv3_1)
    conv3_2 = tfa.layers.InstanceNormalization()(conv3_2)
    pool3 = MaxPool3D(pool_size=(2, 2, 2))(conv3_2)

    #botleneck
    conv4_1 = Conv3D(256, (7, 7, 7), activation=keras.layers.LeakyReLU(alpha=0.3), padding='same')(pool3)
    conv4_2 = Conv3D(256, (7, 7, 7), activation=keras.layers.LeakyReLU(alpha=0.3), padding='same')(conv4_1)

    #decoder
    up1_1 = concatenate([UpSampling3D(size=(2, 2, 2))(conv4_2), conv3_2],axis=-1)
    conv5_1 = Conv3D(128, (7, 7, 7), activation=keras.layers.LeakyReLU(alpha=0.3), padding='same')(up1_1)
    conv5_2 = Conv3D(128, (7, 7, 7), activation=keras.layers.LeakyReLU(alpha=0.3), padding='same')(conv5_1)
    conv5_2 = tfa.layers.InstanceNormalization()(conv5_2)

    up2_1 = concatenate([UpSampling3D(size=(2, 2, 2))(conv5_2), conv2_2],axis=-1)
    conv6_1 = Conv3D(64, (7, 7, 7), activation=keras.layers.LeakyReLU(alpha=0.3), padding='same')(up2_1)
    conv6_2 = Conv3D(64, (7, 7, 7), activation=keras.layers.LeakyReLU(alpha=0.3), padding='same')(conv6_1)

    up3_1 = concatenate([UpSampling3D(size=(2, 2, 2))(conv6_2), conv1_2],axis=-1)
    up3_1 = Dropout(dropout)(up3_1)
    conv7_1 = Conv3D(32, (11, 11, 11), activation=keras.layers.LeakyReLU(alpha=0.3), padding='same')(up3_1)
    conv7_2 = Conv3D(32, (11, 11, 11), activation=keras.layers.LeakyReLU(alpha=0.3), padding='same')(conv7_1)
    conv7_2 = tfa.layers.InstanceNormalization()(conv7_2)

    conv8 = Conv3D(1, (1, 1, 1), activation='sigmoid')(conv7_2)
    #normalization
    conv8 = Lambda(normalizeOutput, name='lambda_normalization')(conv8)

    #make binary mask
    binConv = Lambda(castTensor, name="lambda_cast")(conv8)
    # #defining ROIs
    roiConv = Lambda(ROIs, name="lambda_roi")([input_img, binConv])
    #compute curve
    curve = Lambda(computeCurve, name="lambda_vf")([binConv, roiConv])

    model = tf.keras.models.Model(inputs=input_img, outputs=[conv8, curve])
    opt = tf.keras.optimizers.Adam(lr= learning_rate, decay = learning_decay)
    model.compile(optimizer= opt,loss=[loss_computeCofDistance3D, loss_mae], loss_weights = [0.3, 0.7])

    return model
