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
from sklearn.model_selection import train_test_split
import concurrent.futures
from scipy.stats import skew, kurtosis
from joblib import dump
from joblib import load
from concurrent.futures import ThreadPoolExecutor

start_time = time.time()

def extract_images_from_folder(folder_path):
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']  # Add more extensions if needed
    images = []

    for file_name in os.listdir(folder_path):
        if any(file_name.lower().endswith(ext) for ext in image_extensions):
            images.append(os.path.join(folder_path, file_name))

    return images

# Get the current working directory
current_dir = os.getcwd()

# Relative path to image folder
relative_folder_path = 'attachments\\grayscale'

# Join the current directory with the relative folder path
folder_path = os.path.join(current_dir, relative_folder_path)

images = extract_images_from_folder(folder_path)

#RS analysis

def median_edge_detector(image_matrix):
    image_matrix = image_matrix.astype(np.int64)
    # Create a copy of the image matrix to store the predicted values
    predicted_matrix = image_matrix.copy()

    # Iterate over the image matrix and apply the MED predictor formula
    for i in range(1, image_matrix.shape[0]):
        for j in range(1, image_matrix.shape[1]):
            predicted_matrix[i, j] = np.median([
                image_matrix[i-1, j],
                image_matrix[i, j-1],
                image_matrix[i-1, j] + image_matrix[i, j-1] - image_matrix[i-1, j-1]
            ])

    return predicted_matrix

def calculate_residuals(image_matrix, predicted_matrix):
    # Calculate the residuals
    return image_matrix - predicted_matrix



def calculate_rs_features(residuals):
    # Calculate statistics from the residuals
    features_rs = [np.mean(residuals), np.std(residuals), skew(residuals.flatten()), kurtosis(residuals.flatten())]
    return np.array(features_rs)

def calculate_lbp_features(lbp, num_bins=256):
    # Calculate the histogram of the LBP values
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=num_bins, range=(0, num_bins))
    # Normalize the histogram
    lbp_hist = normalize(lbp_hist.reshape(1, -1), norm='l1')
    return lbp_hist.flatten()



#LBP analysis

def calculate_histogram(lbp):
    # Calculate the histogram of the LBP values
    lbp_hist, _ = np.unique(lbp.ravel(), return_counts=True)
    # Normalize the histogram
    lbp_hist = normalize(lbp_hist.reshape(1, -1), norm='l1')
    return lbp_hist

def calculate_lbp(image_matrix, points=8, radius=1):
    # Calculate the Local Binary Pattern representation of the image
    lbp = feature.local_binary_pattern(image_matrix, points, radius, method="uniform")
    return lbp

#Chi-square attack
def chi_square_attack(image):
    # Calculate the histogram of the image
    histogram = np.histogram(image.flatten(), bins=256, range=(0,256))[0]

    # Calculate the pairs of values
    pairs = np.zeros(128)
    for i in range(0, 256, 2):
        pairs[i // 2] = histogram[i] + histogram[i + 1]

    # Calculate the expected values if no steganography was used
    expected = np.sum(pairs) / 128

    # Calculate the chi-square statistic
    chi_square_stat = np.sum((pairs - expected)**2 / expected)

    return chi_square_stat


#Sample Pair Analysis

def sample_pair_analysis(image):
    # Flatten the image
    image = image.flatten()

    # Calculate the differences between adjacent pixels
    differences = np.diff(image)

    # Count the number of pairs that have the same value and different signs
    same_value = np.sum(differences == 0)
    different_signs = np.sum(np.diff(np.sign(differences)) != 0)

    # Calculate the SPA statistic
    spa_stat = same_value - different_signs

    return spa_stat



from concurrent.futures import ProcessPoolExecutor
from PIL import Image
import numpy as np
import os

# Function to process each image
def process_image(image_path, label):
    # Load the image
    image = Image.open(image_path)
    image = np.array(image.convert('L'))  # Convert to grayscale

    # Calculate the RS Analysis features
    predicted_matrix = median_edge_detector(image)
    residuals = calculate_residuals(image, predicted_matrix)
    features_rs = calculate_rs_features(residuals)

    # Calculate the LBP features
    lbp = calculate_lbp(image)
    features_lbp = calculate_lbp_features(lbp)

    # Calculate the Chi-square attack feature
    features_chi = np.array([chi_square_attack(image)])  # Wrap the scalar in a 1D array

    # Calculate the Sample Pair Analysis feature
    features_spa = np.array([sample_pair_analysis(image)])  # Wrap the scalar in a 1D array

    # Concatenate the features into a single feature vector
    features = np.concatenate((features_rs, features_lbp, features_chi, features_spa))

    return features, label

def fuse_image_features(images, labels):
    # List to store the feature vectors
    features_list = []

    # Create a process pool and process each image in parallel
    with ProcessPoolExecutor() as executor:
        results = executor.map(process_image, images, labels)

    # Add the feature vectors and labels to the lists
    for features, label in results:
        features_list.append(features)

    # Convert the list of feature vectors to a 2D array
    features_array = np.array(features_list)

    return features_array

# for image_path, true_label, predicted_label in zip(all_images, test_labels, test_predictions):
#     print('Image:', image_path)
#     print('True label:', true_label)
#     print('Predicted label:', predicted_label)
#     print()

#Save the model
#dump(clf, 'svm_model.joblib')
end_time = time.time()
execution_time = end_time - start_time
print("Execution Time:", execution_time, "seconds")