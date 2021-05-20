# Mini Project 3
In this project we are required to perform multi-class classification for images of 15 classes (scenes). <br> 
___
# Image representation techniques:
In both approaches in this project, we represent the image as a vector in some vector space. This enables us to utilize famous classification techniques, like KNN and SVMs, which are easily applicable once data is represented in vector format.
___
## Naive Approach: Resize, then unroll
In this approach, we 
* resize an image into a fixed small size (like 16x16)
* Then we unroll the resulting matrix into a vector

Having done that, we can treat every image as a vector. We also did some other operations, like 
* normalizing the resultant vector to unit length, 
* making the small patch have zero mean, 

which have a normalizing effect, and slightly enhances results.
___
## Better approach: Bag of words
A better way to characterize an image is using local features. We have already implemented SIFT-like features, but how can we use them for classification?

We can't possibly match every image with the entire dataset to get which class it belongs to, that is extremely not scalable.

A better approach is to use the Bag of words technique, which is:
* Build a vocabulary set of all possible feature descriptors in the training data
* Then for each training data example, build a histogram, describing the distribution of its own feature descriptors across the vocabulary
* This histogram is then a **feature vector representing the image itself**! Then, we can use simple classification techniques like SVMs and KNN. 

However, if we include all feature descriptors, the data will be huge, and this requires exact matches from the test data, which is unlikely. For that, we can instead cluster this vocabulary into a smaller set of vocabulary. We will then build a histogram for each image of what cluster its features descriptors belong to
___
For the bag of words, we implemented the 2 functions 

`build_vocabulary()`
It takes in
* A list of training images paths
* Vocabulary size (number of clusters)

The function then
* gets the `hog` descriptors of all images and stacks on top of each other
* Clusters the collection of `hog` descriptors, where the number of clusters is the vocabulary size
* Returns the cluster centroids.

`get_bags_of_words()`

This function takes in
* a list of image paths

The function them
* For each image, get its hog features.
* Calculate the distance between each hog feature and all centroids, obtained from the earlier function
* Then, get the histogram of what centroids the descriptors are nearest to.
* Normalize the histogram to unit length
* stack the histograms together, 
* Return the stacked histograms and that represents the combined vectors representing the entire training data

___
# Classifiers
We used both KNN and linear one vs all SVMs for classification. The functions are `svm_classify()` and `nearest_neighbor_classify()`
___
# Hyper-parameters
In this model, there are Classifier hyperparameters like:
* Classifier hyperparameters: (Tolerance or lambda of Linear SVMs)
* How many neighbors to consider in KNN
* Distance metric in KNN (euclidean, manhatten, ....)

There are also non-classifier hyper-parameters like:
* How many clusters to have
* How many iterations for clustering
* "Tolerance" parameter of clustering
* Distance metric of feature descriptor to cluster centroid
* cells per block and pixels per cell parameters for the hog features
* The number of discrete gradient orientations in the hog features

We chose the following parameters:
* KNN => 3 neighbors (using k = 1 makes the model more likely to overfit)
* KNN => euclidean distance (wanted to try chi-squared distance, but there was an error)
* SVM => lambda <= 1 (greater than 1 will reduce test accuracy. We settled on 1)
* For the clusters, we didn't try other values than those written in the comments. Clustering takes a lot of time, and consumes a lot of memory, so we didn't try other configurations, especially when our best accuracy is good enough (70%). So, we do not need further enhancements
* For the distance to clusters, we tried euclidean distance and the Chi-square distance. The Chi-square distance offered a slight improvement in test performance, so, we settled on it.
* For the hog descriptor, we used default number of orientations (9) and used the block and pixel sizes in sift (4x4 cells per block, with 4x4 pixels per cell). Because trying other configurations would mean doing clustering again, we didn't try other configurations
___
# Results
## Tiny Images, with KNN (k = 3)
Accuracy => 23.1%

This poor performance is to be expected, however it is better than a random guess,so an improvement!

## Bag of words, with KNN (k = 3)
Accuracy => 55%

## Bag of words, with linear one vs all SVM (Best accuracy)
Accuracy => 70%
This is by far the best accuracy achieved. Files containing the confusion matrix and detailed summary of the model performance are attached with the project deliverables