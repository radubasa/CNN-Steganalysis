import os
import numpy as np
from PIL import Image
from scipy import stats
from skimage import feature
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import time
import concurrent.futures
from scipy.stats import skew, kurtosis
from joblib import dump, load
from concurrent.futures import ProcessPoolExecutor

# Record the start time for execution time measurement
start_time = time.time()

def extract_images_from_folder(folder_path):
    # Check if the folder exists
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"The folder {folder_path} does not exist.")
    
    # Define the supported image extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
    images = []

    # Iterate over files in the folder and collect image file paths
    for file_name in os.listdir(folder_path):
        if any(file_name.lower().endswith(ext) for ext in image_extensions):
            images.append(os.path.join(folder_path, file_name))

    return images

def median_edge_detector(image_matrix):
    # Convert image matrix to int64 for calculations
    image_matrix = image_matrix.astype(np.int64)
    predicted_matrix = image_matrix.copy()

    # Apply the median edge detector to predict pixel values
    for i in range(1, image_matrix.shape[0]):
        for j in range(1, image_matrix.shape[1]):
            predicted_matrix[i, j] = np.median([
                image_matrix[i-1, j],
                image_matrix[i, j-1],
                image_matrix[i-1, j] + image_matrix[i, j-1] - image_matrix[i-1, j-1]
            ])

    return predicted_matrix

def calculate_residuals(image_matrix, predicted_matrix):
    # Calculate the residuals between the original and predicted matrices
    return image_matrix - predicted_matrix

def calculate_rs_features(residuals):
    # Calculate RS analysis features: mean, standard deviation, skewness, and kurtosis
    features_rs = [np.mean(residuals), np.std(residuals), skew(residuals.flatten()), kurtosis(residuals.flatten())]
    return np.array(features_rs)

def calculate_lbp_features(lbp, num_bins=256):
    # Calculate the histogram of LBP values and normalize it
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=num_bins, range=(0, num_bins))
    lbp_hist = normalize(lbp_hist.reshape(1, -1), norm='l1')
    return lbp_hist.flatten()

def calculate_lbp(image_matrix, points=8, radius=1):
    # Calculate the Local Binary Pattern (LBP) representation of the image
    lbp = feature.local_binary_pattern(image_matrix, points, radius, method="uniform")
    return lbp

def chi_square_attack(image):
    # Calculate the histogram of the image
    histogram = np.histogram(image.flatten(), bins=256, range=(0,256))[0]
    pairs = np.zeros(128)
    for i in range(0, 256, 2):
        pairs[i // 2] = histogram[i] + histogram[i + 1]
    expected = np.sum(pairs) / 128
    # Calculate the chi-square statistic
    chi_square_stat = np.sum((pairs - expected)**2 / expected)
    return chi_square_stat

def sample_pair_analysis(image):
    # Flatten the image and calculate differences between adjacent pixels
    image = image.flatten()
    differences = np.diff(image)
    same_value = np.sum(differences == 0)
    different_signs = np.sum(np.diff(np.sign(differences)) != 0)
    # Calculate the SPA statistic
    spa_stat = same_value - different_signs
    return spa_stat

def process_image(image_path, label):
    # Open the image and convert to grayscale
    image = Image.open(image_path)
    image = np.array(image.convert('L'))

    # Perform RS analysis
    predicted_matrix = median_edge_detector(image)
    residuals = calculate_residuals(image, predicted_matrix)
    features_rs = calculate_rs_features(residuals)

    # Perform LBP analysis
    lbp = calculate_lbp(image)
    features_lbp = calculate_lbp_features(lbp)

    # Perform Chi-square attack
    features_chi = np.array([chi_square_attack(image)])  # Wrap the scalar in a 1D array

    # Perform Sample Pair Analysis
    features_spa = np.array([sample_pair_analysis(image)])  # Wrap the scalar in a 1D array

    # Concatenate all features into a single feature vector
    features = np.concatenate((features_rs, features_lbp, features_chi, features_spa))

    return features, label

def fuse_image_features(images, labels):
    features_list = []
    # Process images in parallel using ProcessPoolExecutor
    with ProcessPoolExecutor() as executor:
        results = executor.map(process_image, images, labels)
    for features, label in results:
        features_list.append((features, label))
    features_array, labels_array = zip(*features_list)
    return np.array(features_array), np.array(labels_array)

def load_images_and_labels(base_folder):
    # Load images and labels from the specified folders
    clean_train = extract_images_from_folder(os.path.join(base_folder, 'train/train/clean'))
    stego_train = extract_images_from_folder(os.path.join(base_folder, 'train/train/stego'))
    clean_val = extract_images_from_folder(os.path.join(base_folder, 'val/val/clean'))
    stego_val = extract_images_from_folder(os.path.join(base_folder, 'val/val/stego'))
    clean_test = extract_images_from_folder(os.path.join(base_folder, 'test/test/clean'))
    stego_test = extract_images_from_folder(os.path.join(base_folder, 'test/test/stego'))

    train_images = clean_train + stego_train
    train_labels = [0] * len(clean_train) + [1] * len(stego_train)
    val_images = clean_val + stego_val
    val_labels = [0] * len(clean_val) + [1] * len(stego_val)
    test_images = clean_test + stego_test
    test_labels = [0] * len(clean_test) + [1] * len(stego_test)

    return train_images, train_labels, val_images, val_labels, test_images, test_labels

def main():
    # Specify the absolute path to the archive folder
    base_folder = 'E:/Scoala/2024/CNN-Steganalysis/CNN-Steganalysis/archive'
    train_images, train_labels, val_images, val_labels, test_images, test_labels = load_images_and_labels(base_folder)

    # Extract and fuse features from images
    train_features, train_labels = fuse_image_features(train_images, train_labels)
    val_features, val_labels = fuse_image_features(val_images, val_labels)
    test_features, test_labels = fuse_image_features(test_images, test_labels)

    # Train an SVM classifier
    clf = svm.SVC(kernel='linear', probability=True)
    clf.fit(train_features, train_labels)

    # Predict and evaluate on validation set
    val_predictions = clf.predict(val_features)
    print("Validation Accuracy:", accuracy_score(val_labels, val_predictions))
    print("Validation Precision:", precision_score(val_labels, val_predictions))
    print("Validation Recall:", recall_score(val_labels, val_predictions))
    print("Validation F1 Score:", f1_score(val_labels, val_predictions))

    # Predict and evaluate on test set
    test_predictions = clf.predict(test_features)
    print("Test Accuracy:", accuracy_score(test_labels, test_predictions))
    print("Test Precision:", precision_score(test_labels, test_predictions))
    print("Test Recall:", recall_score(test_labels, test_predictions))
    print("Test F1 Score:", f1_score(test_labels, test_predictions))

    # Save the trained model
    dump(clf, 'svm_model.joblib')

    # Print the execution time
    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution Time:", execution_time, "seconds")

if __name__ == "__main__":
    main()