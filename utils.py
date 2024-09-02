import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from skimage.measure import find_contours
from skimage.morphology import binary_dilation

def show_images(images, titles=None):
    import matplotlib.pyplot as plt
    n_ims = len(images)
    if titles is None:
        titles = ['(%d)' % i for i in range(1, n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image, title in zip(images, titles):
        a = fig.add_subplot(1, n_ims, n)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    plt.show()

def preprocess_image(img, feature_extraction_method):
    if feature_extraction_method == 0:  # OVERLAPPING_METHOD
        img_copy = img.copy()
        if len(img.shape) > 2:
            img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
        img_copy = cv2.medianBlur(img_copy, 5)
        img_copy = cv2.threshold(img_copy, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        min_vertical, max_vertical = get_corpus_boundaries(img_copy)
        img_copy = img_copy[min_vertical:max_vertical]
        return img_copy

    if feature_extraction_method == 1:  # LINES_METHOD
        img_copy = img.copy()
        if len(img.shape) > 2:
            grayscale_img = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
        else:
            grayscale_img = img.copy()
        img_copy = cv2.threshold(grayscale_img, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        min_vertical, max_vertical = get_corpus_boundaries(img_copy)
        img_copy = img_copy[min_vertical:max_vertical]
        grayscale_img = grayscale_img[min_vertical:max_vertical]
        filter_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        img_copy_sharpened = cv2.filter2D(img_copy, -1, filter_kernel)
        return img_copy_sharpened, grayscale_img

def get_corpus_boundaries(img):
    crop = []
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 1))
    detect_horizontal = cv2.morphologyEx(img, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    contours = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    prev = -1
    for i, c in enumerate(contours):
        if np.abs(prev - int(c[0][0][1])) > 800 or prev == -1:
            crop.append(int(c[0][0][1]))
            prev = int(c[0][0][1])
    crop.sort()
    max_vertical = crop[1] - 20
    min_vertical = crop[0] + 20
    return min_vertical, max_vertical

def segment_image(img, num, grayscale_img=None):
    if grayscale_img is not None:
        grayscale_images = []
        img_copy = np.copy(img)
        kernel = np.ones((1, num))
        img_copy = binary_dilation(img_copy, kernel)
        bounding_boxes = find_contours(img_copy, 0.8)
        for box in bounding_boxes:
            x_min = int(np.min(box[:, 1]))
            x_max = int(np.max(box[:, 1]))
            y_min = int(np.min(box[:, 0]))
            y_max = int(np.max(box[:, 0]))
            if (y_max - y_min) > 50 and (x_max - x_min) > 50:
                grayscale_images.append(grayscale_img[y_min:y_max, x_min:x_max])
        return grayscale_images
    images = []
    img_copy = np.copy(img)
    kernel = np.ones((1, num))
    img_copy = binary_dilation(img_copy, kernel)
    bounding_boxes = find_contours(img_copy, 0.8)
    for box in bounding_boxes:
        x_min = int(np.min(box[:, 1]))
        x_max = int(np.max(box[:, 1]))
        y_min = int(np.min(box[:, 0]))
        y_max = int(np.max(box[:, 0]))
        if (y_max - y_min) > 10 and (x_max - x_min) > 10:
            images.append(img[y_min:y_max, x_min:x_max])
    return images

def overlap_words(words, avg_height):
    overlapped_img = np.zeros((3600, 320))
    index_i = 0
    index_j = 0
    max_height = 0
    for word in words:
        if word.shape[1] + index_j > overlapped_img.shape[1]:
            max_height = 0
            index_j = 0
            index_i += int(avg_height // 2)
        if word.shape[1] < overlapped_img.shape[1] and word.shape[0] < overlapped_img.shape[0]:
            indices = np.copy(overlapped_img[index_i:index_i + word.shape[0], index_j:index_j + word.shape[1]])
            indices = np.maximum(indices, word)
            overlapped_img[index_i:index_i + word.shape[0], index_j:index_j + word.shape[1]] = indices
            index_j += word.shape[1]
            if max_height < word.shape[0]:
                max_height = word.shape[0]
    overlapped_img = overlapped_img[:index_i + int(avg_height // 2), :]
    return overlapped_img

def get_textures(image):
    index_i = 0
    index_j = 0
    texture_size = 100
    textures = []
    while index_i + texture_size < image.shape[0]:
        if index_j + texture_size > image.shape[1]:
            index_j = 0
            index_i += texture_size
        textures.append(np.copy(image[index_i: index_i + texture_size, index_j: index_j + texture_size]))
        index_j += texture_size
    return textures
