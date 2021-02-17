"""
This module contains all processing and anaylsiss operations from the processing
pipeline except fromt he data acquisition phase.
"""

import argparse
import config
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.cluster import KMeans


def normalize(X, a=-1, b=1):
    """
    Normalizes an array to be in range[a, b].

    :param X:   input array
    :param a:   lower bound of target interval
    :param b:   upper bound of target interval

    :return:    X normalized to range between a and b
    """
    X_min = np.min(X)
    X_max = np.max(X)
    X_norm = a + ((X - X_min) * (b - a)) / (X_max - X_min)
    return X_norm


def get_forest_mask(bands):
    """
    Classifies which 64 x 64 patches in the input image correspond to forest
    and which don't. Classification is achieved through a SVM.

    :param bands:   collected bands for input data boint, i.e. B, G, R, NIR

    :return:        coordinates of patches corresponding to non-forest in form
                    of [y_min, x_min, y_max, x_max]
    """
    rgb = get_rgb(bands)
    # set to multiple of 64 -> TODO: make more general
    rgb = cv2.resize(rgb, dsize=(512, 320), interpolation=cv2.INTER_CUBIC)
    height, width, channels = rgb.shape
    # Load SVM from file
    with open(config.SVM_PATH, 'rb') as file:
        svm = pickle.load(file)
        input = []
        num_samples = 0

        for y in range(0, height, 64):
            for x in range(0, width, 64):
                patch = rgb[y:y + 64, x:x + 64]
                input.append(patch)
                num_samples += 1

        input = np.array(input)
        input = input.reshape((num_samples, -1))

        y_predict = svm.predict(input)

        counter = 0
        coords = []
        for y in range(0, height, 64):
            for x in range(0, width, 64):
                if y_predict[counter] == 0:
                    pass
                    # not forest
                    # rgb[y: y + 64, x: x + 64, :] = 0
                    coords.append([y, x, y + 64, x + 64])
                else:
                    # forest
                    pass
                counter += 1

        return coords


def get_bgr(bands):
    """
    Returns the enhanced BGR from the input bands.

    :param bands:   input bands

    :return:        BGR image
    """
    # enhanced bgr
    bgr = np.uint8(3.5 * bands[:, :, 0:3])
    return bgr


def get_rgb(bands):
    """
    Returns the enhanced RGB from the input bands.

    :param bands:   input bands

    :return:        RGB image
    """
    bgr = np.uint8(3.5 * bands[:, :, 0:3])
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


def cluster(img, year):
    """
    Performs image clustering on the RGB image using k-means.

    :param img:     input image as RGB
    :param year:    year of input image
    """
    cv2.imshow(str(year), img)
    Z = img.reshape((-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness, labels, centers = cv2.kmeans(np.float32(Z), 2, None, criteria, 10, flags)

    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    res2 = res.reshape((img.shape))
    cv2.imshow('res2', res2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_NDVI_histogram(bands):
    """
    Computes the histogram of the NDVI image using 3000 bins.

    :param bands:   input bands
    """
    NDVI = np.divide((bands[:, :, 3] - bands[:, :, 2]), (bands[:, :, 3] + bands[:, :, 2]))
    height, width = NDVI.shape
    print("Number of distinct NDVI values: " + str(len(np.unique(NDVI))))

    hist, bin_edges = np.histogram(NDVI, bins=3000)
    plt.bar(bin_edges[:-1], hist)
    plt.xlim(-1, 1)
    # plt.yscale('log')
    plt.title("Histogram of NDVI image")
    plt.show()


def get_NDVI(bands):
    """
    Computes the NDVI image, i.e.  NDVI = (NIR - Red) / (NIR + Red).

    :param bands:   input bands, [g, b, r, NIR]

    :return:        NDVI image in range [-1, 1]
    """

    NDVI = np.divide((bands[:, :, 3] - bands[:, :, 2]), (bands[:, :, 3] + bands[:, :, 2]))
    colormap = mcolors.LinearSegmentedColormap.from_list(
        "my_colormap", ['darkred', 'red', 'orange', 'yellow', 'yellowgreen', 'green'])

    plt.imshow(NDVI, cmap=colormap)
    plt.show()
    return NDVI


def discretize_NDVI(NDVI):
    """
    Discretizes the NDVI image to 4 values.

    :param NDVI:    NDVI image in range [-1, 1]

    :return:        discrtized NDVI image
    """

    NDVI[(NDVI >= -1) & (NDVI <= 0.0)] = 1  # dead plants or other objects
    NDVI[(NDVI > 0) & (NDVI <= 0.33)] = 2  # unhealthy plant
    NDVI[(NDVI > 0.33) & (NDVI <= 0.66)] = 3  # moderately healthy plants, vegetation
    NDVI[(NDVI > 0.66) & (NDVI <= 1.0)] = 4  # very healthy plant, dense vegetation

    colormap = mcolors.LinearSegmentedColormap.from_list(
        "my_colormap", ['darkred', 'yellow', 'green'])

    plt.imshow(NDVI, cmap=colormap)
    plt.show()
    return NDVI


def get_EVI(bands, G=2.5, c1=6.0, c2=7.5, L=1.0):
    """
    Computes the enhanced vegetation index (EVI) using the default parameters.

    :param bands:   bands for input data point, i.e. B, G, R, NIR

    :return:        EVI image
    """
    blue = (bands[:, :, 0] - np.min(bands[:, :, 0])) / \
        (np.max(bands[:, :, 0]) - np.min(bands[:, :, 0]))
    NIR = (bands[:, :, 3] - np.min(bands[:, :, 3])) / \
        (np.max(bands[:, :, 3]) - np.min(bands[:, :, 3]))
    red = (bands[:, :, 2] - np.min(bands[:, :, 2])) / \
        (np.max(bands[:, :, 2]) - np.min(bands[:, :, 2]))
    EVI = G * (NIR - red) / (NIR + (c1 * red) - (c2 * blue) + L)

    return EVI


def discretize_EVI(EVI):
    """
    Discretizes the EVI image to 20 values.

    :param NDVI:    EVI image

    :return:        discrtized EVI image, ranging from 1 to 20
    """
    EVI[(EVI < -1.1)] = 1
    EVI[(EVI < -0.25)] = 2
    EVI[(EVI < -0.1)] = 3
    EVI[(EVI < 0)] = 4
    EVI[(EVI < 0.025)] = 5
    EVI[(EVI < 0.05)] = 6
    EVI[(EVI < 0.075)] = 7
    EVI[(EVI < 0.1)] = 8
    EVI[(EVI < 0.125)] = 9
    EVI[(EVI < 0.15)] = 10
    EVI[(EVI < 0.175)] = 11
    EVI[(EVI < 0.2)] = 12
    EVI[(EVI < 0.25)] = 13
    EVI[(EVI < 0.3)] = 14
    EVI[(EVI < 0.35)] = 15
    EVI[(EVI < 0.4)] = 16
    EVI[(EVI < 0.45)] = 17
    EVI[(EVI < 0.5)] = 18
    EVI[(EVI < 0.55)] = 19
    EVI[(EVI < 0.6)] = 20

    colormap = mcolors.LinearSegmentedColormap.from_list(
        "my_colormap", [[0, 0, 0], [0.75, 0.75, 0.75], [0.86, 0.86, 0.86],
                        [1, 1, 0.88], [1, 0.98, 0.8], [0.93, 0.91, 0.71],
                        [0.87, 0.85, 0.61], [0.8, 0.78, 0.51], [0.74, 0.72, 0.42],
                        [0.69, 0.76, 0.38], [0.64, 0.8, 0.35], [0.57, 0.75, 0.32],
                        [0.5, 0.7, 0.28], [0.44, 0.64, 0.25], [0.38, 0.59, 0.21],
                        [0.31, 0.54, 0.18], [0.25, 0.49, 0.14], [0.19, 0.43, 0.11],
                        [0.13, 0.38, 0.07], [0.06, 0.33, 0.04]])
    plt.imshow(EVI, cmap=colormap)
    plt.show()
    return EVI


def diff_img_analysis(img1, img2, index=None):
    """
    Performas image difference analysis on the vegetation index image simply by
    taking the difference between the two input image. If the result is negative,
    the healthiness of the vegetation improved/vegetation became more dense.
    If the value is positive, the healthiness decreased. We encode no change or
    improvement as black and worsening as white.

    :param img1:    vegetation image 1, first in time
    :param img2:    vegetation image 2, second in time
    :param index:   string of index being used

    :
    """
    if(img1.shape != img2.shape):
        print("Image sizes mismatch. Img1: " + str(img1.shape) + ". Img2: " + str(img2.shape))
        return

    diff = img1 - img2

    if index == 'NDVI':
        # vegetation in this area became worse
        diff_thresh = np.zeros(diff.shape)
        diff_thresh[np.where(diff > 0)] = 1

        # vegetation in this area improved
        diff_thresh[np.where(diff <= 0)] = 0
        return diff_thresh

    elif index == 'EVI':
        # vegetation in this area became worse
        diff_thresh = np.zeros(diff.shape)
        diff_thresh[np.where(diff > 0)] = 1

        # vegetation in this area improved
        diff_thresh[np.where(diff <= 0)] = 0
        return diff_thresh

    else:
        print("Specify a correct vegetation index (NDVI or EVI).")
        return


def NDVI_2_binary(img):
    """
    Filters the discretized NDVI image on the "unheahty vegetation" class.

    :param img: discretized NDVI image
    :return:    binary image
    """
    binary_img = img.copy()
    binary_img[np.where(img == 2.0)] = 1
    binary_img[np.where(img != 2.0)] = 0

    return binary_img


def EVI_2_binary(img):
    """
    Filters the discretized EVI image on the "unheahty vegetation" class.

    :param img: discretized EVI image
    :return:    binary image
    """
    binary_img = img.copy()
    binary_img[:, :] = 0
    binary_img[np.where((img >= 2.0) & (img <= 9.0))] = 1

    return binary_img


def get_cutting_suggestion(img, index=None):
    """
    Computes a cutting suggestion on the unhealthy vegetation area using a
    morphology cascade.

    :param img:     input image
    :param index:   vegetation index being used
    """
    if index == "NDVI":
        img = NDVI_2_binary(img)
    elif index == "EVI":
        img = EVI_2_binary(img)
    else:
        print("Give a proper vegetation index.")
        return

    plt.imshow(img, cmap='gray')
    plt.show()

    # opening: removing small responses
    kernel = np.ones((2, 2), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    plt.imshow(img, cmap='gray')
    plt.show()

    # closing: closing up holes
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    plt.imshow(img, cmap='gray')
    plt.show()

    # dilation: making remaining part larger
    kernel = np.ones((2, 2), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_DILATE, kernel)
    plt.imshow(img, cmap='gray')
    plt.show()


def run_pipeline_morphology(bands):
    """
    Runs the image processing pipeline branch related to cutting suggestion.

    :param bands:  bands corresponding to the image to be processed
    """
    img = get_rgb(bands)
    # show results for forest mask step
    coords = get_forest_mask(bands)

    for y_min, x_min, y_max, x_max in coords:
        img[y_min: y_max, x_min: x_max] = 0

    # show results for forest mask step
    plt.imshow(img)
    plt.show()

    NDVI = get_NDVI(bands)
    NDVI = discretize_NDVI(NDVI)
    # Do we need the EVI here?
    EVI = get_EVI(bands)
    EVI = discretize_EVI(EVI)

    for y_min, x_min, y_max, x_max in coords:
        NDVI[y_min: y_max, x_min: x_max] = 0
        EVI[y_min: y_max, x_min: x_max] = 0

    get_cutting_suggestion(NDVI, index="NDVI")
    get_cutting_suggestion(EVI, index="EVI")


def run_pipeline_change(bands1, bands2):
    """
    Runs the image processing pipeline branch related to change detection.

    :param bands1:  bands corresponding to the first image in time
    :param bands2:  bands corresponding to the second image in time
    """
    img1 = get_rgb(bands1)
    img2 = get_rgb(bands2)
    coords = get_forest_mask(bands1)

    # Pipeline step 2: forest mask
    coords = get_forest_mask(bands1)

    for y_min, x_min, y_max, x_max in coords:
        img1[y_min: y_max, x_min: x_max] = 0
        img2[y_min: y_max, x_min: x_max] = 0

    # show results for forest mask step
    plt.imshow(img1)
    plt.show()
    plt.imshow(img2)
    plt.show()

    # Pipeline step 3: vegetation indices
    # NDVI
    NDVI1 = get_NDVI(bands1)
    NDVI2 = get_NDVI(bands2)
    # discretize to 4 classes
    NDVI1 = discretize_NDVI(NDVI1)
    NDVI2 = discretize_NDVI(NDVI2)

    # EVI
    EVI1 = get_EVI(bands1)
    EVI2 = get_EVI(bands2)
    # discretize to 20 classes
    EVI1 = discretize_EVI(EVI1)
    EVI2 = discretize_EVI(EVI2)

    # set forest mask before computing difference image
    for y_min, x_min, y_max, x_max in coords:
        NDVI1[y_min: y_max, x_min: x_max] = 0
        NDVI2[y_min: y_max, x_min: x_max] = 0
        EVI1[y_min: y_max, x_min: x_max] = 0
        EVI2[y_min: y_max, x_min: x_max] = 0

    diff_thresh_NDVI = diff_img_analysis(NDVI1, NDVI2, index="NDVI")
    plt.imshow(diff_thresh_NDVI, cmap='gray')
    plt.show()

    diff_thresh_EVI = diff_img_analysis(EVI1, EVI2, index="EVI")
    plt.imshow(diff_thresh_EVI, cmap='gray')
    plt.show()


def main(args):
    # bands_2017 = np.load(config.TEST_DATA_2017)
    # forest_masked = get_forest_mask(bands_2017)

    bands_2018 = np.load(config.TEST_DATA_2018)
    bands_2020 = np.load(config.TEST_DATA_2020)

    if args.branch == "IDA":
        run_pipeline_change(bands_2018, bands_2020)
    elif args.branch == "morphology":
        run_pipeline_morphology(bands_2018)
    else:
        print("Invalid pipeline branch.")
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--branch', type=str, help='branch to choose, either IDA or morphology')
    args = parser.parse_args()
    main(args)
