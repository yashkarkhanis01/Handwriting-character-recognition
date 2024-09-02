import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset
import torch.nn.functional as F
from sklearn.svm import SVC
from utils import local_binary_pattern

HISTOGRAM_BINS = 256
NN_LEARNING_RATE = 0.003
NN_WEIGHT_DECAY = 0.01
NN_DROPOUT = 0.25
NN_EPOCHS = 200
NN_BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def model_generator(features, labels, feature_extraction_method, classifier_type):
    histograms = []

    if feature_extraction_method == 0:  # OVERLAPPING_METHOD
        for texture_array in features:
            for texture in texture_array:
                texture = np.uint8(texture)  # Ensure image is of integer type
                lbp = local_binary_pattern(texture, 8, 3, 'default')
                histogram, _ = np.histogram(lbp, density=False, bins=HISTOGRAM_BINS, range=(0, HISTOGRAM_BINS))
                histograms.append(histogram)

    elif feature_extraction_method == 1:  # LINES_METHOD
        for line in features:
            line = np.uint8(line)  # Ensure image is of integer type
            lbp = local_binary_pattern(line, 8, 3, 'default')
            histogram, _ = np.histogram(lbp, density=False, bins=HISTOGRAM_BINS, range=(0, HISTOGRAM_BINS))
            histograms.append(histogram)

    if classifier_type == 0:  # SUPPORT_VECTOR_CLASSIFIER
        model = SVC(kernel='linear')
        model.fit(histograms, labels)
        return model

    if classifier_type == 1:  # NEURAL_NETWORK_CLASSIFIER
        model = nn.Sequential(nn.Linear(HISTOGRAM_BINS, 128),
                              nn.ReLU(),
                              nn.Dropout(p=NN_DROPOUT),
                              nn.Linear(128, 64),
                              nn.ReLU(),
                              nn.Dropout(p=NN_DROPOUT),
                              nn.Linear(64, 3))
        model.to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adamax(model.parameters(), lr=NN_LEARNING_RATE, weight_decay=NN_WEIGHT_DECAY)
        inputs = torch.Tensor(histograms)
        labels = torch.tensor(labels, dtype=torch.long) - 1
        dataset = TensorDataset(inputs, labels)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=NN_BATCH_SIZE, shuffle=True)
        for epoch in range(NN_EPOCHS):
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                output = model(inputs)
                loss = criterion(output, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return model

def predict(model, test_image, feature_extraction_method, classifier_type):
    from utils import preprocess_image, segment_image, overlap_words, get_textures
    if feature_extraction_method == 0:  # OVERLAPPING_METHOD
        img = preprocess_image(test_image)
        words = segment_image(img, 3)
        avg_height = 0
        for word in words:
            avg_height += word.shape[0] / len(words)
        overlapped_img = overlap_words(words, avg_height)
        textures = get_textures(overlapped_img)
        prediction = np.zeros(4)
        for texture in textures:
            texture = np.uint8(texture)  # Ensure image is of integer type
            lbp = local_binary_pattern(texture, 8, 3, 'default')
            histogram, _ = np.histogram(lbp, density=False, bins=HISTOGRAM_BINS, range=(0, HISTOGRAM_BINS))
            if classifier_type == 0:  # SUPPORT_VECTOR_CLASSIFIER
                prediction[model.predict([histogram])] += 1
            if classifier_type == 1:  # NEURAL_NETWORK_CLASSIFIER
                with torch.no_grad():
                    model.eval()
                    histogram = torch.Tensor(histogram)
                    probabilities = F.softmax(model.forward(histogram), dim=0)
                    _, top_class = probabilities.topk(1)
                    prediction[top_class + 1] += 1
        return np.argmax(prediction)

    if feature_extraction_method == 1:  # LINES_METHOD
        img, grayscale_img = preprocess_image(test_image, feature_extraction_method)
        grayscale_lines = segment_image(img, 100, grayscale_img)
        prediction = np.zeros(4)
        for line in grayscale_lines:
            line = np.uint8(line)  # Ensure image is of integer type
            lbp = local_binary_pattern(line, 8, 3, 'default')
            histogram, _ = np.histogram(lbp, density=False, bins=HISTOGRAM_BINS, range=(0, HISTOGRAM_BINS))
            if classifier_type == 0:  # SUPPORT_VECTOR_CLASSIFIER
                prediction[model.predict([histogram])] += 1
            if classifier_type == 1:  # NEURAL_NETWORK_CLASSIFIER
                with torch.no_grad():
                    model.eval()
                    histogram = torch.Tensor(histogram)
                    probabilities = F.softmax(model.forward(histogram), dim=0)
                    _, top_class = probabilities.topk(1)
                    prediction[top_class + 1] += 1
        return np.argmax(prediction)
