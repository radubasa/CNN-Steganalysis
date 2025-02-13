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
from tensorflow.keras import mixed_precision
import os

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

def create_resnet50_model(input_shape):
    # Define a ResNet50 model with pre-trained weights
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    
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
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy'])
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
    train_folder = 'archive/train/train'
    test_folder = 'archive/test/test'
    validation_folder = 'archive/val/val'

    # Model initialization
    model = create_resnet50_model((512, 512, 3))

    # Data augmentation for training data
    train_datagen = ImageDataGenerator(
        rescale=1.0/255.0,
        rotation_range=30,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Data generators for validation and test data (no augmentation, only rescaling)
    val_test_datagen = ImageDataGenerator(rescale=1.0/255.0)

    # Create generators with smaller batch size
    train_generator = train_datagen.flow_from_directory(
        train_folder,
        target_size=(512, 512),
        batch_size=128,  # Decreased batch size
        class_mode='categorical'
    )

    validation_generator = val_test_datagen.flow_from_directory(
        validation_folder,
        target_size=(512, 512),
        batch_size=128,  # Decreased batch size
        class_mode='categorical'
    )

    # Create the ResNet50 model
    input_shape = (512, 512, 3)
    model = create_resnet50_model(input_shape)

    # Learning rate reduction and early stopping
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Define the path to save the best model
    best_model_path = 'best_model.keras'

    # Create the ModelCheckpoint callback
    model_checkpoint = ModelCheckpoint(best_model_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

    # Compute class weights to handle class imbalance
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_generator.classes), y=train_generator.classes)
    class_weights = {i: class_weights[i] for i in range(len(class_weights))}

    # Train the model with additional options
    history = model.fit(
        train_generator,
        epochs=25,
        validation_data=validation_generator,
        class_weight=class_weights,
        callbacks=[reduce_lr, early_stopping, model_checkpoint]
    )

    # Save the final model
    model.save('model/modelResNet50.keras')

    # Define the test generator
    test_generator = val_test_datagen.flow_from_directory(
        test_folder,
        target_size=(512, 512),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )

    # Evaluate the model
    y_pred = model.predict(test_generator)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = test_generator.classes[:len(y_pred_classes)]

    # Print the metrics
    print("ResNet50 Classification Report:")
    print(classification_report(y_true, y_pred_classes))

    print("ResNet50 Confusion Matrix:")
    conf_matrix = confusion_matrix(y_true, y_pred_classes)
    print(conf_matrix)

    # Accuracy
    accuracy = np.mean(y_pred_classes == y_true)
    print("ResNet50 Accuracy:", accuracy)

    # Precision
    precision = precision_score(y_true, y_pred_classes, average='binary')
    print("ResNet50 Precision:", precision)

    # Recall
    recall = recall_score(y_true, y_pred_classes, average='binary')
    print("ResNet50 Recall:", recall)

    # F1 Score
    f1 = f1_score(y_true, y_pred_classes, average='binary')
    print("ResNet50 F1 Score:", f1)

    # Area Under the Receiver Operating Characteristic curve (AUC-ROC)
    roc_auc = roc_auc_score(y_true, y_pred_classes)
    print("ResNet50 AUC-ROC:", roc_auc)

if __name__ == "__main__":
    main()