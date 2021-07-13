#Vascular Input Extraction (VIF) deep learning model
import numpy as np
import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2" #GPU to be used

import glob
import scipy.io
import pandas as pd

import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from utils_vif import *
from model_vif import *

def inference_mode(args):

    print('Loading data')
    volume_data = np.load(args.input_path)

    print('Preprocessing')
    vol_pre = preprocessing(volume_data)

    print('Loading model')
    model = unet3d(img_size = (120, 120, 120, 7),\
                     learning_rate = 1e-3,\
                     learning_decay = 1e-9)

    model.load_weights(args.model_weight_path)

    print('Prediction')
    y_pred_mask, y_pred_vf = model.predict(vol_pre)
    y_pred_mask = y_pred_mask > 0.5
    y_pred_mask = y_pred_mask.astype(float)

    print('Resizing volume (padding)')
    y_pred_mask_rz = resize_mask(y_pred_mask)#padding

    return  y_pred_vf, y_pred_mask_rz

def training_model(args):

    print("Tensorflow", tf.__version__)
    print("Keras", keras.__version__)

    DATASET_DIR = args.dataset_path
    train_set = load_data(os.path.join(DATASET_DIR,"train/images"))
    val_set = load_data(os.path.join(DATASET_DIR,"val/images"))
    test_set = load_data(os.path.join(DATASET_DIR,"test/images"))

    print('Training')

    print("Train:", len(train_set))
    print("Val:", len(val_set))
    print("Test:", len(test_set))

    model = unet3d(img_size = (120, 120, 120, 7),\
                     learning_rate = 1e-6,\
                     learning_decay = 1e-9)

    batch_size = args.batch_size
    train_gen = train_generator(os.path.join(DATASET_DIR,"train/"), train_set, batch_size)
    val_gen = train_generator(os.path.join(DATASET_DIR,"val/"), val_set, batch_size)

    model_path = os.path.join(args.save_checkpoint_path,'model_weight.h5')

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, min_lr=1e-15)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=40)
    save_model = tf.keras.callbacks.ModelCheckpoint(model_path, verbose=0, monitor='val_loss', save_best_only=True)
    callbackscallbac  = [save_model, reduce_lr, early_stop]

    print('Training')
    history = model.fit(
    train_gen,
    steps_per_epoch=len(train_set)/batch_size,
    epochs=args.epochs,
    validation_data = val_gen,
    validation_steps=len(val_set)/batch_size,
    callbacks = callbackscallbac)

    np.save(os.path.join(args.save_checkpoint_path,'history.npy'), history.history)
    plot_history(os.path.join(args.save_checkpoint_path,'history.npy'), os.path.join(args.save_checkpoint_path,'history.png'))

    print("End")

def evaluate_model(args):

    model = unet3d(img_size = (120, 120, 120, 7),\
                     learning_rate = 1e-3,\
                     learning_decay = 1e-9)

    model.load_weights(args.model_weight_path)

    data = load_data(os.path.join(args.input_folder,"images"))
    print('Number of images:', len(data))
    batch_size = 1
    table = []
    mae_array = []
    dist_array = []

    for i in range(len(data)):
        print('Image:', data[i])
        gen_ = train_generator_physical(args.input_folder, [data[i]], batch_size, data_augmentation=False)
        batch_img, batch_label = next(gen_)# with cropping

        y = np.load(os.path.join(args.input_folder,"images_newShape",data[i]))#mask without cropping

        y_pred_mask, y_pred_vf = model.predict(batch_img)
        y_pred_mask = y_pred_mask > 0.5
        y_pred_mask = y_pred_mask.astype(float)

        mae = loss_mae_without_factor(batch_label[1].astype(np.float32), y_pred_vf.astype(np.float32))
        d_distance = loss_computeCofDistance3D(batch_label[0], y_pred_mask.reshape(1,120, 120, 120,1))
        table.append([data[i], d_distance.numpy()[0], (mae.numpy()[0]),])

        mae_array.append(mae.numpy()[0])
        dist_array.append(d_distance.numpy()[0])

        if args.save_image == 1:
            plt.figure(figsize=(15,5), dpi=250)
            plt.subplot(1,2,1)
            plt.title('Vascular Function (VF):'+data[i])
            x = np.arange(len(y_pred_vf[0]))
            plt.yticks(fontsize=19)
            plt.xticks(fontsize=19)
            plt.plot(x, y_pred_vf[0], 'r', label='Auto', lw=3)
            plt.plot(x, (batch_label[1])[0], 'b', label='Manual', lw=3)
            plt.legend(loc="upper right", fontsize=16)
            plt.savefig(os.path.join(args.save_output_path, data[i]+'.png'), bbox_inches="tight")
            plt.close()

    mae_array = np.asarray(mae_array)
    dist_array = np.asarray(dist_array)

    df = pd.DataFrame(table, columns =['Id','Distance (delta)','MAE'])
    df.to_csv(os.path.join(args.save_output_path, 'results.csv'))

if __name__== "__main__":

    parser = argparse.ArgumentParser(description="VIF model")
    parser.add_argument("--mode", type=str, default="inference", help="training mode (training) or inference mode (inference) or evaluate mode (eval)")
    parser.add_argument("--dataset_path", type=str, default=" ", help="path to dataset")
    parser.add_argument("--save_output_path", type=str, default=" ", help="path to save model's checkpoint")
    parser.add_argument("--save_checkpoint_path", type=str, default=" ", help="path to save model's checkpoint")
    parser.add_argument("--model_weight_path", type=str, default=" ", help="file of the model's checkpoint")
    parser.add_argument("--input_path", type=str, default=" ", help="input image path")
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
    parser.add_argument("--loss_weights", type=float, default=[0.3, 0.7], help="loss weights for spatial information and temporal information")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--input_folder", type=str, default=" ", help="path of the folder to be evaluated")
    parser.add_argument("--save_image", type=int, default=0, help="save the vascular function as image")

    args = parser.parse_args()

    if args.mode == "inference":
        print('Mode:', args.mode)
        _, mask = inference_mode(args)
        np.save(args.save_output_path+'/mask.npy', mask)
        scipy.io.savemat(args.save_output_path+'/mask.mat',{'mask_pred':mask})
    elif args.mode == "training":
        print('Mode:', args.mode)
        training_model(args)
    elif args.mode == "eval":
        print('Mode:', args.mode)
        evaluate_model(args)
    else:
        print('Error: mode not found!')
