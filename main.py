import time
from data import read_random_images, extract_features
from model import model_generator, predict


epochs = 100
root = 'D:\\Sem 4\\data'
feature_extraction_method = 0  # OVERLAPPING_METHOD
classifier_type = 0  # SUPPORT_VECTOR_CLASSIFIER
correct_predictions = 0
total_execution_time = 0

for epoch in range(epochs):
    images, labels, test_image, test_label = read_random_images(root)
    start_time = time.time()
    features, features_labels = extract_features(images, labels, feature_extraction_method)
    model = model_generator(features, features_labels, feature_extraction_method, classifier_type)
    prediction = predict(model, test_image, feature_extraction_method, classifier_type)
    execution_time = time.time() - start_time
    total_execution_time += execution_time
    if prediction == test_label:
        correct_predictions += 1
    print("Epoch #{} | Execution time {} seconds | Model accuracy {}".format(epoch + 1, round(execution_time, 2), round((correct_predictions / (epoch + 1)) * 100, 2)))

print("Model accuracy = {}% using {} sample tests.".format((correct_predictions / epochs) * 100, epochs))
print("Total execution time = {} using {} sample tests.".format(round(total_execution_time, 2), epochs))
