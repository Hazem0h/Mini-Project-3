import numpy as np
import matplotlib
from skimage.io import imread
from skimage.color import rgb2grey
from skimage.feature import hog
from skimage.transform import resize
from scipy.spatial.distance import cdist
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.svm import LinearSVC

# global variables
N_CELLS = 4
N_PIXELS = 4

def get_tiny_images(image_paths):
    """
    This feature is inspired by the simple tiny images used as features in
    80 million tiny images: a large dataset for non-parametric object and
    scene recognition. A. Torralba, R. Fergus, W. T. Freeman. IEEE
    Transactions on Pattern Analysis and Machine Intelligence, vol.30(11),
    pp. 1958-1970, 2008. http://groups.csail.mit.edu/vision/TinyImages/

    Inputs:
        image_paths: a 1-D Python list of strings. Each string is a complete
                     path to an image on the filesystem.
    Outputs:
        An n x d numpy array where n is the number of images and d is the
        length of the tiny image representation vector. e.g. if the images
        are resized to 16x16, then d is 16 * 16 = 256.

    To build a tiny image feature, resize the original image to a very small
    square resolution (e.g. 16x16). You can either resize the images to square
    while ignoring their aspect ratio, or you can crop the images into squares
    first and then resize evenly. Normalizing these tiny images will increase
    performance modestly.

    As you may recall from class, naively downsizing an image can cause
    aliasing artifacts that may throw off your comparisons. See the docs for
    skimage.transform.resize for details:
    http://scikit-image.org/docs/dev/api/skimage.transform.html#skimage.transform.resize

    Suggested functions: skimage.transform.resize, skimage.color.rgb2grey,
                         skimage.io.imread, np.reshape
    """
    # define some helpful constants
    RESCALE_SIZE = 16
    n = len(image_paths)
    d = RESCALE_SIZE ** 2
    # initialize the output array
    all_data = np.empty((n, d))
    # loop over all image paths
    for index, path in enumerate(image_paths):
        # read the image
        image = imread(path, as_gray=True)
        # reshape the image
        rescaled_image = resize(image, output_shape = (RESCALE_SIZE, RESCALE_SIZE), anti_aliasing=True)
        # unroll into 1D vector
        unrolled = np.reshape(rescaled_image, newshape = (RESCALE_SIZE**2,))
        # zero mean
        unrolled = unrolled - unrolled.mean()
        # normalize to unit length
        unrolled = unrolled / (np.sqrt(np.sum(np.square(unrolled))))
        # add to the output array
        all_data[index] = unrolled
        
    return all_data


def build_vocabulary(image_paths, vocab_size):
    """
    This function should sample HOG descriptors from the training images,
    cluster them with kmeans, and then return the cluster centers.

    Inputs:
        image_paths: a Python list of image path strings
         vocab_size: an integer indicating the number of words desired for the
                     bag of words vocab set

    Outputs:
        a vocab_size x (z*z*9) (see below) array which contains the cluster
        centers that result from the K Means clustering.

    You'll need to generate HOG features using the skimage.feature.hog() function.
    The documentation is available here:
    http://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.hog

    However, the documentation is a bit confusing, so we will highlight some
    important arguments to consider:
        cells_per_block: The hog function breaks the image into evenly-sized
            blocks, which are further broken down into cells, each made of
            pixels_per_cell pixels (see below). Setting this parameter tells the
            function how many cells to include in each block. This is a tuple of
            width and height. Your SIFT implementation, which had a total of
            16 cells, was equivalent to setting this argument to (4,4).
        pixels_per_cell: This controls the width and height of each cell
            (in pixels). Like cells_per_block, it is a tuple. In your SIFT
            implementation, each cell was 4 pixels by 4 pixels, so (4,4).
        feature_vector: This argument is a boolean which tells the function
            what shape it should use for the return array. When set to True,
            it returns one long array. We recommend setting it to True and
            reshaping the result rather than working with the default value,
            as it is very confusing.

    It is up to you to choose your cells per block and pixels per cell. Choose
    values that generate reasonably-sized feature vectors and produce good
    classification results. For each cell, HOG produces a histogram (feature
    vector) of length 9. We want one feature vector per block. To do this we
    can append the histograms for each cell together. Let's say you set
    cells_per_block = (z,z). This means that the length of your feature vector
    for the block will be z*z*9.

    With feature_vector=True, hog() will return one long np array containing every
    cell histogram concatenated end to end. We want to break this up into a
    list of (z*z*9) block feature vectors. We can do this using a really nifty numpy
    function. When using np.reshape, you can set the length of one dimension to
    -1, which tells numpy to make this dimension as big as it needs to be to
    accomodate to reshape all of the data based on the other dimensions. So if
    we want to break our long np array (long_boi) into rows of z*z*9 feature
    vectors we can use small_bois = long_boi.reshape(-1, z*z*9).

    The number of feature vectors that come from this reshape is dependent on
    the size of the image you give to hog(). It will fit as many blocks as it
    can on the image. You can choose to resize (or crop) each image to a consistent size
    (therefore creating the same number of feature vectors per image), or you
    can find feature vectors in the original sized image.

    ONE MORE THING
    If we returned all the features we found as our vocabulary, we would have an
    absolutely massive vocabulary. That would make matching inefficient AND
    inaccurate! So we use K Means clustering to find a much smaller (vocab_size)
    number of representative points. We recommend using sklearn.cluster.KMeans
    to do this. Note that this can take a VERY LONG TIME to complete (upwards
    of ten minutes for large numbers of features and large max_iter), so set
    the max_iter argument to something low (we used 100) and be patient. You
    may also find success setting the "tol" argument (see documentation for
    details)
    """

    # initialize all hog features
    features_list = []
    # loop over all images and extract hog features
    print("Getting HOG features")
    for path in image_paths:
        image = imread(path, as_gray=True)
        image_hog_vectors = hog(image, cells_per_block = (N_CELLS, N_CELLS), pixels_per_cell= (N_PIXELS, N_PIXELS), feature_vector= True)

        # each row is the feature vectorssss of a block (interest point)
        image_hog_vectors = image_hog_vectors.reshape(-1, N_CELLS*N_CELLS*9)

        # append to all hog features
        features_list.append(image_hog_vectors)

    # now stack them all vertically
    print("Stacking HOG features")
    all_hog_features = np.vstack(features_list)

    print("Clustering")
    # now cluster them using kmeans ########################################(tol ????)
    CluseterObj = MiniBatchKMeans(n_clusters = vocab_size, max_iter= 100)
    CluseterObj.fit(all_hog_features)
    centroids = CluseterObj.cluster_centers_

    return centroids


def get_bags_of_words(image_paths):
    """
    This function should take in a list of image paths and calculate a bag of
    words histogram for each image, then return those histograms in an array.

    Inputs:
        image_paths: A Python list of strings, where each string is a complete
                     path to one image on the disk.

    Outputs:
        An nxd numpy matrix, where n is the number of images in image_paths and
        d is size of the histogram built for each image.

    Use the same hog function to extract feature vectors as before (see
    build_vocabulary). It is important that you use the same hog settings for
    both build_vocabulary and get_bags_of_words! Otherwise, you will end up
    with different feature representations between your vocab and your test
    images, and you won't be able to match anything at all!

    After getting the feature vectors for an image, you will build up a
    histogram that represents what words are contained within the image.
    For each feature, find the closest vocab word, then add 1 to the histogram
    at the index of that word. For example, if the closest vector in the vocab
    is the 103rd word, then you should add 1 to the 103rd histogram bin. Your
    histogram should have as many bins as there are vocabulary words.

    Suggested functions: scipy.spatial.distance.cdist, np.argsort,
                         np.linalg.norm, skimage.feature.hog
    """

    vocab = np.load('vocab.npy')
    print('Loaded vocab from file.')
    
    n_clusters = vocab.shape[0]
    bags_of_words = np.zeros((len(image_paths), n_clusters))
    # loop over each image
    for example_index, path in enumerate(image_paths):
        if example_index% 10 == 0:
            print("Bag of words example", example_index)
        image = imread(path, as_gray=True)

        # get the hog features of the image
        hog_features = hog(image, pixels_per_cell=(N_PIXELS, N_PIXELS), cells_per_block=(N_CELLS, N_CELLS), feature_vector=True)
        hog_features = hog_features.reshape(-1, N_CELLS * N_CELLS * 9)

        # get the distance between each hog feature and the centroid (Chi-square distance)
        distances = cdist(hog_features, vocab, lambda x,y: 0.5*np.sum(np.square(x-y)/(x+y)))
        nearest_cluster_indices = np.argsort(distances, axis = 1)[:,0]

        # iniitalize the histogram
        hist = np.zeros((n_clusters,))
        for cluster_index in nearest_cluster_indices:
            hist[cluster_index] += 1

        # normalize the histogram to unit length
        hist = hist / np.sqrt(np.sum(np.square(hist)))
        
        # assign the histogram to its proper location 
        bags_of_words[example_index] = hist

    print("Done building bags of words")
    return bags_of_words


def svm_classify(train_image_feats, train_labels, test_image_feats):
    """
    This function will predict a category for every test image by training
    15 many-versus-one linear SVM classifiers on the training data, then
    using those learned classifiers on the testing data.

    Inputs:
        train_image_feats: An nxd numpy array, where n is the number of training
                           examples, and d is the image descriptor vector size.
        train_labels: An nx1 Python list containing the corresponding ground
                      truth labels for the training data.
        test_image_feats: An mxd numpy array, where m is the number of test
                          images and d is the image descriptor vector size.

    Outputs:
        An mx1 numpy array of strings, where each string is the predicted label
        for the corresponding image in test_image_feats

    We suggest you look at the sklearn.svm module, including the LinearSVC
    class. With the right arguments, you can get a 15-class SVM as described
    above in just one call! Be sure to read the documentation carefully.
    """

    #multi_class='crammer_singer' #another method but we will use ovr
    multi_class='ovr' # one versus rest
    
    clf = LinearSVC(penalty = 'l2', loss = 'squared_hinge',
                                multi_class = multi_class, intercept_scaling = 1, class_weight = None,
                                verbose = 0, random_state = None,
                                tol = 1)
    out = clf.fit(train_image_feats, train_labels)
    
    prediction = out.predict(test_image_feats)
    
    return prediction


def nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats, k = 3, dist = "euclidean"):
    """
    This function will predict the category for every test image by finding
    the training image with most similar features. You will complete the given
    partial implementation of k-nearest-neighbors such that for any arbitrary
    k, your algorithm finds the closest k neighbors and then votes among them
    to find the most common category and returns that as its prediction.

    Inputs:
        train_image_feats: An nxd numpy array, where n is the number of training
                           examples, and d is the image descriptor vector size.
        train_labels: An nx1 Python list containing the corresponding ground
                      truth labels for the training data.
        test_image_feats: An mxd numpy array, where m is the number of test
                          images and d is the image descriptor vector size.

    Outputs:
        An mx1 numpy list of strings, where each string is the predicted label
        for the corresponding image in test_image_feats

    The simplest implementation of k-nearest-neighbors gives an even vote to
    all k neighbors found - that is, each neighbor in category A counts as one
    vote for category A, and the result returned is equivalent to finding the
    mode of the categories of the k nearest neighbors. A more advanced version
    uses weighted votes where closer matches matter more strongly than far ones.
    This is not required, but may increase performance.

    Be aware that increasing k does not always improve performance - even
    values of k may require tie-breaking which could cause the classifier to
    arbitrarily pick the wrong class in the case of an even split in votes.
    Additionally, past a certain threshold the classifier is considering so
    many neighbors that it may expand beyond the local area of logical matches
    and get so many garbage votes from a different category that it mislabels
    the data. Play around with a few values and see what changes.

    Useful functions:
        scipy.spatial.distance.cdist, np.argsort, scipy.stats.mode
    """

    print("Calssifying using KNN")
    # Gets the distance between each test image feature and each train image feature
    distances = cdist(test_image_feats, train_image_feats,  dist)

    # Get the indeces of the sorted the distances across each row
    sorted_indeces = np.argsort(distances, axis = 1)

    # get the indeces of the nearest kth points for each row (test image)
    k_nearest_indeces = sorted_indeces[:, :k]
    
    # get the corresponding labels (map each index to its label, per row), (use list() for eager evaluation)
    # initiate the mapping function and the labels list
    mapping_function = lambda x: train_labels[x]
    labels = []

    # loop over each point, and get the labels of the nearest k points
    for row in k_nearest_indeces:
        mapped_labels = list(map(mapping_function, row))
        labels.append(mapped_labels)

    # get the most occuring label, per row (get the element with maximum count, per row)
    most_occuring_labels = [max(row, key = row.count) for row in labels]
    return np.array(most_occuring_labels)
