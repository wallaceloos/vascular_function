import os
import numpy as np
import random
import matplotlib.pyplot as plt

def preprocessing(vol):

    #original dimension: (256, 240, 120)
    #cropping dimension: (120, 120, 120)

    batch_images = np.empty((1, 120, 120, 120, 7))
    vol_crop = np.zeros([120,120,120,7])
    vol = (vol-np.min(vol))/((np.max(vol)-np.min(vol)))
    vol_crop = vol[102:(256-34), 60:(240-60),:,:]
    batch_images[0] = vol_crop

    return batch_images


def resize_mask(mask):

    mask_rz = np.zeros([mask.shape[0], 256, 240, 120], dtype=float)
    mask_rz[:,102:(256-34), 60:(240-60),:] = mask[:,:,:,:,0]

    return mask_rz

def load_data(path):

    filesList = [f for f in os.listdir(path)]
    return np.asarray(filesList)

def shift_vol(vol, mask):
    new_vol = np.zeros(vol.shape)
    new_mask = np.zeros(mask.shape)

    shift_horizontal = np.random.randint(low=10, high=15, size=1)[0]
    direction = np.random.randint(2, size=1)[0]

    if direction:
        new_vol[:,0:vol.shape[1]-shift_horizontal, :, :] = vol[:,shift_horizontal:vol.shape[1], :, :]
        new_mask[:,0:vol.shape[1]-shift_horizontal, :] = mask[:,shift_horizontal:vol.shape[1], :]
    else:
        new_vol[:,shift_horizontal:vol.shape[1], :, :] = vol[:,0:vol.shape[1]-shift_horizontal, :, :]
        new_mask[:,shift_horizontal:vol.shape[1], :] = mask[:,0:vol.shape[1]-shift_horizontal, :]

    shift_vertical = np.random.randint(low=10, high=15, size=1)[0]
    new_vol[0:new_vol.shape[0]-shift_vertical, :, :, :] = new_vol[shift_vertical:vol.shape[0], :, :, :]
    new_mask[0:new_mask.shape[0]-shift_vertical, :, :] = new_mask[shift_vertical:vol.shape[0],: , :]

    return new_vol, new_mask

def train_generator(DATASET_DIR, data_set, batch_size = 1, temporal_res = 7, data_augmentation = True, shuffle = True):

    batch_images = np.empty((batch_size, 120, 120, 120, temporal_res))
    batch_masks = np.empty((batch_size, 120, 120, 120,1))
    batch_curve = np.empty((batch_size, temporal_res))
    batch_cof = np.empty((batch_size, 3))

    while True:
      for i in range(batch_size):
        if shuffle == True:
          name_id = random.randint(0, len(data_set)-1)
        else:
          name_id = 0

        path_img = data_set[name_id]
        vol = np.load(DATASET_DIR+"/images/"+path_img)
        vol_crop = np.zeros([120,120,120,temporal_res])
        #normalization
        vol = (vol-np.min(vol))/((np.max(vol)-np.min(vol)))
        mask = np.load(DATASET_DIR+"/masks/"+path_img)

        #data augmentation
        if data_augmentation:
            vol, mask = shift_vol(vol, mask)

        #cropping
        vol_crop= vol[102:(256-34), 60:(240-60),:,:]

        #cropping mask
        mask_crop = np.zeros([120,120,120])
        mask_crop = mask[102:(256-34), 60:(240-60),:]

        #True VF
        mask_train_ = mask_crop.reshape(120,120,120,1)
        roi_ = vol_crop*mask_train_
        num = np.sum(roi_, axis = (0,1,2), keepdims=False)
        den = np.sum(mask_train_, axis = (0,1,2), keepdims=False)
        intensities = num/(den+1e-8)
        intensities = np.asarray(intensities)

        #CoM
        ii, jj, kk = np.meshgrid(np.arange(120), np.arange(120), np.arange(120), indexing='ij')
        ii = ii.astype(np.float32)
        jj = jj.astype(np.float32)
        kk = kk.astype(np.float32)

        xx = ii*mask_crop
        yy = jj*mask_crop
        zz = kk*mask_crop

        xx = np.sum(xx).astype(np.float32)
        yy = np.sum(yy).astype(np.float32)
        zz = np.sum(zz).astype(np.float32)

        total = np.sum(mask_crop)
        total = total.astype(np.float32)
        #-----------------------------------------------------------------------

        batch_images[i] = vol_crop
        batch_masks[i] = mask_crop.reshape(120, 120, 120, 1)
        batch_curve[i] = intensities
        batch_cof[i] = np.array([float(xx/(total+1e-10)), float(yy/(total+1e-10)), float(zz/(total+1e-10))])

      yield batch_images, [batch_cof, batch_curve]

def plot_history(path, save_path):

    history = np.load(path, allow_pickle=True).item()
    for key in history.keys():
        print (key)

    plt.figure(figsize=(14,5), dpi=350)
    plt.subplot(1,3,1)
    plt.grid('on')
    plt.title('Total loss')
    plt.plot(history['loss'], 'b', lw=2, alpha=0.7, label='Training')
    plt.plot(history['val_loss'], 'r', lw=2, alpha=0.7, label='Val')
    plt.legend(loc="upper right")

    plt.subplot(1,3,2)
    plt.title('MAE')
    plt.grid('on')
    plt.plot(history['lambda_vf_loss'], 'b', lw=2, alpha=0.7, label='Training')
    plt.plot(history['val_lambda_vf_loss'], 'r', lw=2, alpha=0.7, label='Val')
    plt.legend(loc="upper right")

    plt.subplot(1,3,3)
    plt.title('CoM')
    plt.grid('on')
    plt.title('Distance')
    plt.plot(history['lambda_normalization_loss'], 'b', lw=2, alpha=0.7, label='Training')
    plt.plot(history['val_lambda_normalization_loss'], 'r', lw=2, alpha=0.7, label='Val')
    plt.legend(loc="upper right")

    plt.savefig(save_path, bbox_inches="tight")
