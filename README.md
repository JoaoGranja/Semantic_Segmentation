# Semantic Segmentation

In this project, a step by step process will be presented for the Image Semantic Segmentation task on Oxford-IIIT pet dataset. Image segmentation is the task of classifying each pixel in an image from a predefined set of classes. So objects pixels of the image which belongs to the same class shall be classifyed as so. Semantic segmentation can have many application such on medical images (segmenting a tumor), autonomous vehicles (detecting drivable regions) and satellite image analysis (land mapping).


In this Image Semantic Segmentation project, 2 notebooks were built to train the models over the Oxford-IIIT pet dataset. On 'semantic_segmentation_main.ipynb' notebook, the models are build from scratch. Instead, on 'keras_segmentation.ipynb' notebook, the models are imported from the [image-segmentation-keras](https://github.com/divamgupta/image-segmentation-keras) repository. This last notebook was mainly usefull for understanding how to handle and approach the image semantic segmentation task so it will be only provided details for 'semantic_segmentation_main.ipynb' notebook.


There are several different models architectures to apply for this task. However almost of of them use the encoder-decoder structure where the encoder where it outputs a tensor containing information about the objects, and its shape and size. Then the decoder takes this information and produces the segmentation maps. The decoder model can be a pre-trained model like VGG, Resnet, mobilenet, mobileNetV2. In this project, a light encoder model is applied so it will be mobilenet or mobileNetV2. The overall model architectures used on this project are:

*   Unet
*   FCN
*   DeepLabv3
*   Pspnet
*   Segnet

All of these models are coded on a separate module (models). When we are training a model for a semantic segmentation task, it is usefull to compute the Intersection over Union (IoU). It is a number from 0 to 1 that specifies the amount of overlap between the predicted and ground truth mask.
* an IoU of 0 means that there is no overlap between the masks
* an IoU of 1 means that the union of the maaks is the same as their overlap indicating that they are completely overlapping

After training the models over the training dataset, the fine-tuning techniques is applied to improve the model accuracy.
I will compare the performance of all models looking on loss, accuracy and IoU metrics over the testing dataset.

Oxford-IIIT pet dataset will be used by directly importing it from tensorflow dataset API. This dataset is a 37 category pet image dataset with roughly 200 images for each class. The images have large variations in scale, pose and lighting. All images have an associated ground truth annotation of breed. More information is available on [Oxford-IIT dataset homepage](https://www.tensorflow.org/datasets/catalog/oxford_iiit_pet)


The 'semantic_segmentation_main.ipynb' script was created to train and evaluate the models over the Oxford-IIIT pet dataset. In this project, I took the following steps:

<ul>
  <li>Colab preparation - In this step,  all necessary packages/libraries are installed and my google drive account is shared.</li>
  <li>Configuration and Imports - All modules are imported and the 'args' dictionary is built with some configuration parameters. </li>
  <li>Loading the dataset - Oxford-IIIT dataset is loaded and the data are analysed. </li>
  <li>Data pre-processing and data augmentation - Resize all images to the same size, prepare the training and testing batches and perform some data augmentation. </li>
  <li>Optimizer - Choose the optimizer for model training </li>
  <li>Metric - Define the metric to computer the IoU </li>
  <li>Model - Based on 'args' configuration, make the model. The model architecture is built on models module. </li>
  <li>Training - The training process runs in this step. Several callbacks are used to improve the trainig process. </li>
  <li>Visualize models result - After the model is trained, the accuracy, IoU and loss of the model are plotted.</li>
  <li>Evaluation - After all models are trained, the evaluation over a testing dataset is done. </li>
</ul>

In sume, the six models achieved

of accuracy over the testing dataset, with VGG16 model achieved the highest score (more than 83%).
The goal of this project was not to achieve the highest accuracy score over the CIFAR100 dataset but to approach the Image Classification task. For better results, several future works are usefull (use better data augmentation techniques, apply regularization techniques to reduce overfitting, tune hyperparameters of the model) 