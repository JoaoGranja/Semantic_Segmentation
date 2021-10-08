# Semantic Segmentation

## **Project Description**
In this project, a step by step process will be presented for the Image Semantic Segmentation task on [Oxford-IIT pet dataset](https://www.tensorflow.org/datasets/catalog/oxford_iiit_pet). Image segmentation is the task of classifying each pixel in an image from a predefined set of classes. So objects pixels of the image which belongs to the same class shall be classifyed as so. Semantic segmentation can have many application such on medical images (segmenting a tumor), autonomous vehicles (detecting drivable regions) and satellite image analysis (land mapping).

## **Notebooks**
In this Image Semantic Segmentation project, 2 notebooks were built to train the models over the [Oxford-IIT pet dataset](https://www.tensorflow.org/datasets/catalog/oxford_iiit_pet). In [semantic_segmentation_main.ipynb](https://github.com/JoaoGranja/Semantic_Segmentation/blob/master/semantic_segmentation_main.ipynb) notebook, the models are built from scratch. Instead, in [keras_segmentation.ipynb](https://github.com/JoaoGranja/Semantic_Segmentation/blob/master/keras_segmentation_main.ipynb) notebook, the models are imported from the [image-segmentation-keras](https://github.com/divamgupta/image-segmentation-keras) repository. This last notebook was mainly usefull for understanding how to handle and approach the image semantic segmentation task so it will be only provided details for the [semantic_segmentation_main.ipynb](https://github.com/JoaoGranja/Semantic_Segmentation/blob/master/semantic_segmentation_main.ipynb) notebook.

## **Dataset**
Oxford-IIIT pet dataset will be used by directly importing it from tensorflow dataset API. This dataset is a 37 category pet image dataset with roughly 200 images for each class. The images have large variations in scale, pose and lighting. All images have an associated ground truth annotation of breed. More information is available on [Oxford-IIT dataset homepage](https://www.tensorflow.org/datasets/catalog/oxford_iiit_pet)

## **Models Architectures**
There are several different models architectures to apply for this task. However almost of them use the encoder-decoder structure where the encoder outputs a tensor containing information about the objects, and its shape and size. Then the decoder takes this information and produces the segmentation maps. The encoder part can be a pre-trained model like vgg, resnet, mobilenet, mobileNetV2. In this project, a light encoder model is applied so it will be mobilenet or mobileNetV2. The overall model architectures used on this project are:

*   Unet
*   FCN
*   DeepLabv3
*   Pspnet
*   Segnet

## **IoU Metric**
All of these models are defined and coded on a separate module (models). When we are training a model for a semantic segmentation task, it is usefull to compute the Intersection over Union (IoU) so a specific metric is defined on this project. IoU is a number from 0 to 1 that specifies the amount of overlap between the predicted and ground truth mask.
* an IoU of 0 means that there is no overlap between the masks
* an IoU of 1 means that the union of the maaks is the same as their overlap indicating that they are completely overlapping

## **Project steps**
The [semantic_segmentation_main.ipynb](https://github.com/JoaoGranja/Semantic_Segmentation/blob/master/semantic_segmentation_main.ipynb) script was created to train and evaluate the models over the Oxford-IIIT pet dataset. In this project, I took the following steps:

<ul>
  <li><strong>Colab preparation</strong> - In this step,  all necessary packages/libraries are installed and my google drive account is shared.</li>
  <li><strong>Configuration and Imports</strong> - All modules are imported and the 'args' dictionary is built with some configuration parameters. </li>
  <li><strong>Loading the dataset</strong> - Oxford-IIIT dataset is loaded and the data are analysed. </li>
  <li><strong>Data pre-processing and data augmentation</strong> - Resize all images to the same size, prepare the training and testing batches and perform some data augmentation. </li>
  <li><strong>Optimizer</strong> - Choose the optimizer for model training </li>
  <li><strong>Metric</strong> - Define the metric to computer the IoU </li>
  <li><strong>Model</strong> - Based on 'args' configuration, make the model. The model architecture is built on models module. </li>
  <li><strong>Training</strong> - The training process runs in this step. Several callbacks are used to improve the trainig process. </li>
  <li><strong>Visualize models result</strong> - After the model is trained, the accuracy, IoU and loss of the model are plotted.</li>
  <li><strong>Evaluation</strong> - After all models are trained, the evaluation over a testing dataset is done. </li>
</ul>

After training the models over the training dataset, the fine-tuning technique is also applied to improve the model accuracy. On the fine tuning step, the encoder part of the model is also trained to force the weights of the encoder to be tuned from generic feature maps to features associated specifically with the dataset. After training all models, I will compare the performance of them looking on loss, accuracy and IoU metrics over the testing dataset.

## **Models Results**
In sume, five models: *mobileNetV2_Unet*, *mobilenet_fcn_8*, *mobileNetV2_pspnet*, *mobileNetV2_segnet* and *Deeplabv3*, were trained. Comparing the results of these models over the testing dataset, it is possible to conclude that the model with the best result is *mobileNetV2_Unet* followed by *mobilenet_fcn_8*. Indeed *mobileNetV2_Unet* has the highest accuracy, around **91%**, and IoU, around **0.75**, evaluated on testing dataset. The second best model, *mobilenet_fcn_8* achieved **87%** of accuracy and **0.65** of IoU. 

## **Conclusion**
The goal of this project was not to achieve the highest accuracy/IoU score over the [Oxford-IIT pet dataset](https://www.tensorflow.org/datasets/catalog/oxford_iiit_pet) but to approach the Image Semantic Segmentation task. For better results, several future works can be usefull (use better data augmentation techniques, apply regularization techniques to reduce overfitting, tune hyperparameters of the model, use deeper encoder model,...) 
