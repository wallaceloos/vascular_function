This is a TensorFlow implementation for the Vascular Function Extraction Model (VFEM). A pretrained model is also provided.  
Paper: [Extraction of a vascular function for a fully automated dynamic contrast-enhanced magnetic resonance brain image processing pipeline](https://onlinelibrary.wiley.com/doi/abs/10.1002/mrm.29054)
### Requirements

 - TensorFlow 2.3 
 - Keras 2.4
 - Python 3.6 
 - Numpy 
 - Scipy
 - Pandas

### Preparing the data

<p align="justify">The data was saved in numpy format following the radiological orientation. Its original dimension is 256 x 240 x 120. Then the volume was undersampled using the bolus arrival information and cropped to fit into the dimensions 120 x 120 x 120 x 7. Since a region over the transverse sinus was used as the vascular function, it is important to make sure that the transverse sinus is preserved. Empirically, this cropping is performing removing equally, 25% of the original dimension of each side. And 14% of the original dimension at the bottom, and at the top 39% of the original dimension. After this step, the data was normalized using the min-max normalization. This data preparation is performed by our data generator, which is also provided. A shift operation is performed during the training for data augmentation.
</div>

### Inference

To use the model you can load the weights provided [here](https://uofc-my.sharepoint.com/:u:/r/personal/wallace_souzaloos_ucalgary_ca/Documents/model_weight_vf/0307.h5?csf=1&web=1&e=PNq5jm) and run:

    python main_vif.py --mode inference --input_path /path/to/data/input_data.npy \
    --model_weight_path /path/to/model_weight/weight.h5  \
    --save_output_path /path/to/folder/output/

A sample from the dataset can be download [here](https://uofc-my.sharepoint.com/:f:/g/personal/wallace_souzaloos_ucalgary_ca/Egus2uREswlOidCIwCf99wwBwED4lmWavcNNc370oSow6g?e=cs8lmH).
<p align="justify">The model will predict a vascular function and a 3D mask. Because the original data was undersampled, the predicted vascular function will only have the number of points that was undersampled. Please, use the 3D mask predicted over the original data to estimate a new vascular function. The pretrained model is using the following weight loss: 0.3 and 0.7. These weights were the ones in which the model achieved the best results.

### Training
In order to train the model, please organize your data set as follows
```
dataset
├── train
│   ├──images
│        └── id_x.npy
│   ├── masks
│        └── id_x.npy
├── val
│   ├──images
│        └── id_x.npy
│   ├── masks
│        └── id_x.npy
├── test
│   ├──images
│        └── id_x.npy
│   ├── masks
│        └── id_x.npy
```
To train the model you can run:

    python main_vif.py --mode training --dataset_path /path/to/dataset/ \
    --save_checkpoint_path  /path/to/save/save_weight/


### Evaluating Model

To evaluate the model you can run:  
 
    python main_vif.py --mode eval --input_folder /path/to/data/folder/ \
    --model_weight_path  /path/to/model/weight.h5 --save_output_path /path/to/folder/to/save/results/
  

