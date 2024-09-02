import os
import random
from os import walk
import cv2
from utils import preprocess_image, segment_image, overlap_words, get_textures

AVAILABLE_WRITERS = 672

def read_random_images(root):
    images = []
    labels = []
    test_images = []
    test_labels = []
    for i in range(3):
        found_images = False
        while not found_images:
            images_path = root
            random_writer = random.randrange(AVAILABLE_WRITERS)
            if random_writer < 10:
                random_writer = "00" + str(random_writer)
            elif random_writer < 100:
                random_writer = "0" + str(random_writer)
            images_path = os.path.join(images_path, str(random_writer))
            if not os.path.isdir(images_path):
                continue
            _, _, filenames = next(walk(images_path))
            if len(filenames) <= 2 and i == 2 and len(test_images) == 0:
                continue
            if len(filenames) >= 2:
                found_images = True
                chosen_filenames = []
                for j in range(2):
                    random_filename = random.choice(filenames)
                    while random_filename in chosen_filenames:
                        random_filename = random.choice(filenames)
                    chosen_filenames.append(random_filename)
                    images.append(cv2.imread(os.path.join(images_path, random_filename)))
                    labels.append(i + 1)
                if len(filenames) >= 3:
                    random_filename = random.choice(filenames)
                    while random_filename in chosen_filenames:
                        random_filename = random.choice(filenames)
                    chosen_filenames.append(random_filename)
                    test_images.append(cv2.imread(os.path.join(images_path, random_filename)))
                    test_labels.append(i + 1)
    test_choice = random.randint(0, len(test_images) - 1)
    test_image = test_images[test_choice]
    test_label = test_labels[test_choice]
    return images, labels, test_image, test_label

def extract_features(images, labels, feature_extraction_method):
    if feature_extraction_method == 1:  # LINES_METHOD
        lines_labels = []
        lines = []
        for image, label in zip(images, labels):
            image, grayscale_image = preprocess_image(image, feature_extraction_method)
            grayscale_lines = segment_image(image, 100, grayscale_image)
            for line in grayscale_lines:
                lines.append(line)
                lines_labels.append(label)
        return lines, lines_labels

    if feature_extraction_method == 0:  # OVERLAPPING_METHOD
        textures = []
        textures_labels = []
        for image, label in zip(images, labels):
            image = preprocess_image(image, feature_extraction_method)
            words = segment_image(image, 3)
            avg_height = 0
            for word in words:
                avg_height += word.shape[0] / len(words)
            overlapped_img = overlap_words(words, avg_height)
            new_textures = get_textures(overlapped_img)
            textures.append(new_textures)
            for j in range(len(new_textures)):
                textures_labels.append(label)
        return textures, textures_labels
