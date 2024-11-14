import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix, recall_score, f1_score, precision_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
import os
from concurrent.futures import ThreadPoolExecutor

def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = np.array(image).reshape(512, 512, 3)
    return image

def preprocess_data(image_paths, labels):
    # Load images and convert to RGB
    images = [preprocess_image(path) for path in image_paths]
    images = np.array(images).reshape(-1, 512, 512, 3)  # Reshape to (num_images, 512, 512, 3)

    # Apply StandardScaler to each image individually
    scaler = StandardScaler()
    images_scaled = np.zeros_like(images, dtype=np.float32)
    for i in range(images.shape[0]):
        for j in range(3):  # Apply scaler to each channel separately
            images_scaled[i, :, :, j] = scaler.fit_transform(images[i, :, :, j])

    # Convert labels to categorical
    labels = to_categorical(labels, num_classes=2)

    return images_scaled, labels, scaler

def create_cnn_model(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Conv2D(256, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        Dropout(0.5),
        Dense(2, activation='softmax')  # 2 output classes
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_resnet50_model(input_shape):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(2, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Freeze the layers of the base model
    for layer in base_model.layers:
        layer.trainable = False
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def load_dataset(folder):
    clean_folder = os.path.join(folder, 'clean')
    stego_folder = os.path.join(folder, 'stego')
    clean_images = [os.path.join(clean_folder, f) for f in os.listdir(clean_folder) if f.lower().endswith('.png')]
    stego_images = [os.path.join(stego_folder, f) for f in os.listdir(stego_folder) if f.lower().endswith('.png')]

    image_paths = clean_images + stego_images
    labels = [0] * len(clean_images) + [1] * len(stego_images)

    return image_paths, labels

def create_tf_dataset(image_paths, labels, batch_size=32, buffer_size=1000):
    def load_and_preprocess_image(path, label):
        image = tf.numpy_function(preprocess_image, [path], tf.float32)
        image.set_shape((512, 512, 3))
        return image, label

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

def main():
    # Paths to datasets
    # Define the paths to the datasets
    train_folder = 'e:/Scoala/2024/CNN-Steganalysis/CNN-Steganalysis/archive/train/train'
    test_folder = 'e:/Scoala/2024/CNN-Steganalysis/CNN-Steganalysis/archive/test/test'
    validation_folder = 'e:/Scoala/2024/CNN-Steganalysis/CNN-Steganalysis/archive/val/val'
    # train_folder = 'CNN-Steganalysis/archive/train/train'
    # test_folder = 'CNN-Steganalysis/archive/test/test'
    # validation_folder = 'CNN-Steganalysis/archive/val/val'

    # Load and preprocess training data
    train_image_paths, train_labels = load_dataset(train_folder)
    validation_image_paths, validation_labels = load_dataset(validation_folder)
    test_image_paths, test_labels = load_dataset(test_folder)

    # Create TensorFlow datasets
    train_dataset = create_tf_dataset(train_image_paths, train_labels)
    validation_dataset = create_tf_dataset(validation_image_paths, validation_labels)
    test_dataset = create_tf_dataset(test_image_paths, test_labels)

    # Create the CNN model
    input_shape = (512, 512, 3)
    model = create_cnn_model(input_shape)  # Comment this line to use ResNet50
    #model = create_resnet50_model(input_shape)  # Uncomment this line to use ResNet50

    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.2,
        preprocessing_function=preprocess_input  # Use ResNet50 preprocessing
    )

    # Learning rate reduction and early stopping
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Define the path to save the best model
    best_model_path = 'best_model.keras'

    
    # Create the ModelCheckpoint callback
    model_checkpoint = ModelCheckpoint(best_model_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

    # Compute class weights
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights = {i: class_weights[i] for i in range(len(class_weights))}

    # Train the model
    history = model.fit(
        datagen.flow_from_directory(train_folder, target_size=(512, 512), batch_size=128, class_mode='categorical'),
        epochs=9,
        validation_data=datagen.flow_from_directory(validation_folder, target_size=(512, 512), batch_size=64, class_mode='categorical'),
        class_weight=class_weights,
        callbacks=[reduce_lr, early_stopping, model_checkpoint]
    )

    model.save('model/modelCNN.keras')

    # Evaluate the model (using the last iteration of the model)
    y_pred = model.predict(test_dataset)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.array([label for _, label in test_dataset.unbatch()])  # Convert test labels to 1D array

    # Print the metrics
    print("CNN Classification Report:")
    print(classification_report(y_true, y_pred_classes))

    print("CNN Confusion Matrix:")
    conf_matrix = confusion_matrix(y_true, y_pred_classes)
    print(conf_matrix)

    # Accuracy
    accuracy = np.mean(y_pred_classes == y_true)
    print("CNN Accuracy:", accuracy)

    # Precision
    precision = precision_score(y_true, y_pred_classes)
    print("CNN Precision:", precision)

    # Recall
    recall = recall_score(y_true, y_pred_classes)
    print("CNN Recall:", recall)

    # F1 Score
    f1 = f1_score(y_true, y_pred_classes)
    print("CNN F1 Score:", f1)

    # Specificity
    tn, fp, fn, tp = conf_matrix.ravel()
    specificity = tn / (tn + fp)
    print("CNN Specificity:", specificity)

    # False Positive Rate (FPR)
    fpr = fp / (fp + tn)
    print("CNN False Positive Rate (FPR):", fpr)

    # Area Under the Receiver Operating Characteristic curve (AUC-ROC)
    roc_auc = roc_auc_score(y_true, y_pred_classes)
    print("CNN AUC-ROC:", roc_auc)

    # Confusion Matrix details
    print("True Positives (TP):", tp)
    print("True Negatives (TN):", tn)
    print("False Positives (FP):", fp)
    print("False Negatives (FN):", fn)

if __name__ == "__main__":
    main()