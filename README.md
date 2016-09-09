# DeepLearningSTL-10

##TRAINING ON UNLABELED STL-10 DATA:

**STL-10 Dataset**

STL-10 dataset is an Image recognition dataset for developing unsupervised feature learning,deep learning,self-taught learning algorithms. STL-10 dataset basically consist of many more unlabeled dataset as compared to labeled dataset.It is inspired by CIFAR-10 dataset but has fewer training examples and very large collection of unlabeled data images. This dataset was basically designed
to exploit unlabeled data for image classication. Training a network on unlabeled data acts as a useful prior for image classification. After learning features through unlabeled data the labeled training data is used for ne tuning the weights which results in better accuracy for image classication by a network as will be shown by our experiment.

**Data Overview**

STL-10 Dataset:-

Images Resolution : 96 X 96

500 training images(10 pre-dened folds),800 test images per class.

100000 unlabeled images for unsupervised learning.

10 classes:airplane, bird, car, cat, deer, horse, monkey, ship, truck.

**Model**

**Preprocessing Unlabeled Data**

STL-10 datset provides us with a huge collection of unlabeled data(100000), but due to memeory restrictions this data cannot be uploaded on the memory as a whole so to circumvent this problem we are processing the unlabeled data in batches,
Batch Size : 5000

The data provided by the dataset has each image of dimension:3 x 96 x 96. 
The images are present in RGB format, for image classication we need to overcome various effects produced by different lighting condition or noises and certain other factors that may cause our network to miss-classify same images into different categories.

**Normalization of Images**

The RGB channels are rst converted to YUV channel before further processing of data can be carried out,this conversion of channels helps in reducing noises that otherwise would have been an hinderance for image classification. After conversion into YUV channels the Y channel is normalized using Spatial ContrastiveNormalization, and other channels(U and V) are normalized with
the mean and standard deviation of dataset currently inthe batch, the mean
of the batch is subtracted from each image and then then it is divided by the
obtained standard deviation of the batch.

**Training on unlabeled data**

In real world we have large sets of unlabeled data as compared to labeled data.Therefore, it is really essential to learn from unlabeled dataset to tackle thereal-world problems. In STL-10 we are doing semi-supervised training in which we first extract the patches from unlabeled data set and train our network on it using k-means clustering algorithm. The centroids obtained from this clustering are fine tuned using supervised training data to better represent the features in an image.

**Patch Extraction**

Before k means clustering algorithm can be applied to the unlabeled data we divide the input image into patches,this process is known as patch-extraction. In patch-extraction we take random patches from the input images, these patches represents different features present in an image like vertical edges, horizontal edges, color blobs etc.The combination of these features are useful for identification of the image.

Calculations:

Patch-size : 9 X 9

Stride : Patches are being taken from random location 

Total Patches : 87 patches/image

Total Number of images:100000

Total Patches: 100000 X 87 =8700000

**K-Means Clustering**

After Patch-Extraction process, extracted patches are sent to k-means clustering algorithm.The k-means clustering algorithm clusters the patches that represent the same features in an image, for example-all the patches from the input that represent horizontal edges come under same clusters,in this way every different feature that constitute an image forms a cluster and k-means algorithm provides us with centroids of clusters representing dierent features that constitute an image.

Calculations:

Number of Centroids : 64

Centroid Dimensions : 64 x 243

The features obtained by k-means clustering forms the first layer of our image classication network and is further ne-tuned with the help of supervised training on labeled data for better accuracy in classication of an input image.

**Supervised Network**

Our Supervised netwrok consist of 7 convolution layers with Max-pooling Layers in between. The schema showing our supervised network is explained below.

**Data Augmentation**

Since, the supervised data is very less,its a good idea to augment the training data set.Data augmentation is useful as it produces noises in the data and helps the network to better classify the images. In our network we are using horizontal flip and rotation of image for training data augmentation, we use 10 percent rotation of few of our images of training data.

**First layer**

First layer of our network comprises of the features obtained from k-means clustering algorithm,the weights of the first convolution layer are initialised with the clusters obtained from clutering patches of unlabeled data which are then further fine-tuned using supervised learning.

Calculation:

Input image Dimensions : 3 X 96 X 96

Number of input planes : 3

Number of output planes : 64

Receptive Field : 9 X 9

Strides:1

Total number of weight parameter : 64 X 3 X 9 X 9

Note:K-means centroids are reshaped from 64 X 243 to 64 X 3 X 9 X 9

**Max-pooling**

The first convolution layer is followed by max-pooling layer,the max pooling layer applies the max pooling function over the obtained filters which helps in reducing the number of parameters that should be passed to the next layer.

Calculation:

Receptive field :4 X 4

Strides = 4

**Second Layer:**

Second layer of our network also comprises of combination of Convolution and max-pooling layer,the second layer further identifies various high-level features present in the input image.

Calculations:

Input planes : 64

Output planes: 128

Receptive eld :9 X 9

Strides : 1

**Third Layer**

Third layer consist of a convolution layer followed by a max-pooling layer,the combination of these two is used to further identify the image for different features obtained from previous layer,every layer identies the high level features from the images obtained from previous layer.

Calculations:

Input planes : 128

Output planes : 256

Receptive eld : 9 X 9

Strides : 1

Max-Pooling Layer Calculations:

Input Image : 256 X 6 X 6

Receptive eld :3 X 3

Strides : 1

**Non-Linear Layer**

The non-linear function applied in all of the above convolutional modules is RELU. RELU has an inherent property of non-saturation which makes it an excellent candidate for non-lineartity Classifier The output from the Fourth layer of convolution network is forwarded to the linear classier which consist of six layers.

Linear Classifier:

[input --> (1) --> (2) --> (3) --> (4) --> (5) --> (6) --> output]

(1):nn.Dropout(0.500000)

(2):nn.Linear(4608 --> 512)

(3):nn.BatchNormalization

(4):nn.ReLU

(5):nn.Dropout(0.500000)

(6): nn.Linear(512 --> 10)

The output from the Linear Classifier classifies the image into 10 classes.

**Results:**

Number of iterations : 1000

Learning rate :1

Learning rate decay : 1e-7

Accuracy obtained on the Training Data :77.32

Accuracy obtained on validation set :74

