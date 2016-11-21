# USAGE
# python compare.py

# import the necessary packages
from skimage.measure import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
import operator
import logging

def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    #  sum of the squared difference between the two images;
    #  NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    # return the MSE, the lower the error, the more "similar" the two images are
    return err



def compare_histo(imageA, imageB):
    h1=cv2.calcHist([imageA], [0], None, [16], [0, 256])
    h2= cv2.calcHist([imageB], [0], None, [16], [0, 256])
    sim = math.sqrt(reduce(operator.add, list(map(lambda a, b: (a - b) ** 2, h1, h2))) / len(h1))
    return sim

def compare_AKAZE_Features(imageA, imageB):
    # compare akaze features descriptor
    # f1 = cv2.cvtColor(imageA[int(np.round(imageB.shape[0] * 0.75)):,
    #                   int(np.round(imageB.shape[0] * 0.4)):int(np.round(imageB.shape[0] * 0.6))], cv2.COLOR_BGR2GRAY)
    # f2 = cv2.cvtColor(imageB[int(np.round(imageB.shape[0] * 0.75)):,
    #                   int(np.round(imageB.shape[0] * 0.4)):int(np.round(imageB.shape[0] * 0.6))], cv2.COLOR_BGR2GRAY)
    extractor = cv2.AKAZE_create()
    kp2, desc2 = extractor.detectAndCompute(imageA, None)
    kp1, desc1 = extractor.detectAndCompute(imageB, None)

    if desc1 is not None and desc2 is not None:
        logging.info('desc1:' + str(desc1.shape[0]) + ' desc2:' + str(desc2.shape[0]))
    return np.abs(desc1.shape[0] - desc2.shape[0])


def compare_images(imageA, imageB, title):
    # compute the mean squared error and structural similarity
    # index for the images
    m = mse(imageA, imageB)
    s = ssim(imageA, imageB)
    sim = compare_histo(imageA, imageB)
    akaze = compare_AKAZE_Features(imageA, imageB)

    # setup the figure
    fig = plt.figure(title)
    plt.suptitle("MSE: %.2f, SSIM: %.2f, sim histo: %.2f, akaze: %.2f" % (m, s, sim, akaze))

    # show first image
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(imageA, cmap = plt.cm.gray)
    plt.axis("off")

    # show the second image
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(imageB, cmap = plt.cm.gray)
    plt.axis("off")

    # show the images
    plt.show()




# load the images -- the original, the original + contrast,
# and the original + photoshop
# original = cv2.imread("images/jp_gates_original.png")
# contrast = cv2.imread("images/jp_gates_contrast.png")
# shopped = cv2.imread("images/jp_gates_photoshopped.png")

original = cv2.imread("images/meteo1.jpg")
contrast = cv2.imread("images/meteo2.jpg")
shopped = cv2.imread("images/jp_gates_photoshopped.png")


# convert the images to grayscale
original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
contrast = cv2.cvtColor(contrast, cv2.COLOR_BGR2GRAY)
shopped = cv2.cvtColor(shopped, cv2.COLOR_BGR2GRAY)

# initialize the figure
fig = plt.figure("Images")
images = ("Original", original), ("Contrast", contrast), ("Photoshopped", shopped)

# loop over the images
for (i, (name, image)) in enumerate(images):
    # show the image
    ax = fig.add_subplot(1, 3, i + 1)
    ax.set_title(name)
    plt.imshow(image, cmap = plt.cm.gray)
    plt.axis("off")

# show the figure
plt.show()

# compare the images
#compare_images(original, original, "Original vs. Original")
compare_images(original, contrast, "Original vs. Contrast")
compare_images(original, shopped, "Original vs. Photoshopped")