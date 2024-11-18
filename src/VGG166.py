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
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
import os
from concurrent.futures import ThreadPoolExecutor

def preprocess_image(image_path):
    # Load image and convert to RGB
    image = Image.open(image_path).convert('RGB')
    # Resize image to 512x512 and convert to numpy array
    image = np.array(image).reshape(512, 512, 3)
    return image.astype(np.float32)

def preprocess_data(image_paths, labels):
    # Load and preprocess images
    images = [preprocess_image(path) for path in image_paths]
    images = np.array(images).reshape(-1, 512, 512, 3)  # Reshape to (num_images, 512, 512, 3)

    # Apply StandardScaler to each image individually
    scaler = StandardScaler()
    images_scaled = np.zeros_like(images, dtype=np.float32)
    for i in range(images.shape[0]):
        for j in range(3):  # Apply scaler to each channel separately
            images_scaled[i, :, :, j] = scaler.fit_transform(images[i, :, :, j])

    # Convert labels to categorical (one-hot encoding)
    labels = to_categorical(labels, num_classes=2)

    return images_scaled, labels, scaler

def create_vgg16_model(input_shape):
    # Define a VGG16 model with pre-trained weights
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Unfreeze the top layers of the base model
    for layer in base_model.layers[-4:]:
        layer.trainable = True
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = Dropout(0.5)(x)
    predictions = Dense(2, activation='softmax')(x)  # Adjusted for 2 classes
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile the model with Adam optimizer and categorical cross-entropy loss
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def load_dataset(folder):
    # Load image paths and labels from the dataset folder
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

    # Create a TensorFlow dataset from image paths and labels
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

def main():
    # Paths to datasets
    train_folder = 'e:/Scoala/2024/CNN-Steganalysis/CNN-Steganalysis/archive/train/train'
    test_folder = 'e:/Scoala/2024/CNN-Steganalysis/CNN-Steganalysis/archive/test/test'
    validation_folder = 'e:/Scoala/2024/CNN-Steganalysis/CNN-Steganalysis/archive/val/val'

    # Load and preprocess training data
    train_image_paths, train_labels = load_dataset(train_folder)
    validation_image_paths, validation_labels = load_dataset(validation_folder)
    test_image_paths, test_labels = load_dataset(test_folder)

    # Create TensorFlow datasets
    train_dataset = create_tf_dataset(train_image_paths, train_labels)
    validation_dataset = create_tf_dataset(validation_image_paths, validation_labels)
    test_dataset = create_tf_dataset(test_image_paths, test_labels)

    # Create the VGG16 model
    input_shape = (512, 512, 3)
    model = create_vgg16_model(input_shape)  # Use VGG16 model

    # Data augmentation
    datagen = ImageDataGenerator(
        rescale=1.0/255.0,  # Normalize pixel values to [0, 1]
        rotation_range=30,  # Randomly rotate images by up to 30 degrees
        width_shift_range=0.3,  # Randomly shift images horizontally by up to 30%
        height_shift_range=0.3,  # Randomly shift images vertically by up to 30%
        shear_range=0.3,  # Randomly shear images by up to 30%
        zoom_range=0.3,  # Randomly zoom into images by up to 30%
        horizontal_flip=True,  # Randomly flip images horizontally
        fill_mode='nearest'  # Fill in missing pixels with the nearest value
    )

    # Learning rate reduction and early stopping
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Define the path to save the best model
    best_model_path = 'best_model.keras'

    # Create the ModelCheckpoint callback
    model_checkpoint = ModelCheckpoint(best_model_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

    # Compute class weights to handle class imbalance
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights = {i: class_weights[i] for i in range(len(class_weights))}

    # Train the model
    history = model.fit(
        datagen.flow_from_directory(train_folder, target_size=(512, 512), batch_size=64, class_mode='categorical'),
        epochs=50,  # Train for more epochs
        validation_data=datagen.flow_from_directory(validation_folder, target_size=(512, 512), batch_size=32, class_mode='categorical'),
        class_weight=class_weights,
        callbacks=[reduce_lr, early_stopping, model_checkpoint]
    )

    # Save the final model
    model.save('model/modelVGG16.keras')

    # Evaluate the model (using the last iteration of the model)
    test_generator = datagen.flow_from_directory(test_folder, target_size=(512, 512), batch_size=64, class_mode='categorical', shuffle=False)
    y_pred = model.predict(test_generator)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = test_generator.classes  # Use the classes attribute from the generator

    # Print the metrics
    print("VGG16 Classification Report:")
    print(classification_report(y_true, y_pred_classes))

    print("VGG16 Confusion Matrix:")
    conf_matrix = confusion_matrix(y_true, y_pred_classes)
    print(conf_matrix)

    # Accuracy
    accuracy = np.mean(y_pred_classes == y_true)
    print("VGG16 Accuracy:", accuracy)

    # Precision
    precision = precision_score(y_true, y_pred_classes, average='binary')
    print("VGG16 Precision:", precision)

    # Recall
    recall = recall_score(y_true, y_pred_classes, average='binary')
    print("VGG16 Recall:", recall)

    # F1 Score
    f1 = f1_score(y_true, y_pred_classes, average='binary')
    print("VGG16 F1 Score:", f1)

    # # Specificity
    # tn, fp, fn, tp = conf_matrix.ravel()
    # specificity = tn / (tn + fp)
    # print("VGG16 Specificity:", specificity)

    # # False Positive Rate (FPR)
    # fpr = fp / (fp + tn)
    # print("VGG16 False Positive Rate (FPR):", fpr)

    # Area Under the Receiver Operating Characteristic curve (AUC-ROC)
    roc_auc = roc_auc_score(y_true, y_pred_classes)
    print("VGG16 AUC-ROC:", roc_auc)

    # Confusion Matrix details
    # print("True Positives (TP):", tp)
    # print("True Negatives (TN):", tn)
    # print("False Positives (FP):", fp)
    # print("False Negatives (FN):", fn)

if __name__ == "__main__":
    main()