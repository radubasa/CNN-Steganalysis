import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D, DepthwiseConv2D, Input, Concatenate
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
import keras_tuner as kt

def preprocess_image(image_path):
    # Load image and convert to RGB
    image = Image.open(image_path).convert('RGB')
    # Resize image to 512x512 and convert to numpy array
    image = np.array(image).reshape(512, 512, 3)
    # Normalize using ImageNet mean and std
    image = image / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    return (image - mean) / std

def preprocess_data(image_paths, labels):
    # Load and preprocess images
    images = [preprocess_image(path) for path in image_paths]
    images = np.array(images).reshape(-1, 512, 512, 3)  # Reshape to (num_images, 512, 512, 3)
    return images, labels

# Custom High-Pass Filter Layer
def HighPassLayer():
    kernel_init = tf.constant_initializer([[[[-1]], [[2]], [[-1]]],
                                           [[[2]], [[-4]], [[2]]],
                                           [[[-1]], [[2]], [[-1]]]])
    return tf.keras.layers.Conv2D(filters=1, kernel_size=(3,3), padding='same',
                                  kernel_initializer=kernel_init, trainable=True)

def build_model(hp):
    inputs = Input(shape=(512, 512, 3))
    x = HighPassLayer()(inputs)

    for i in range(hp.Int('conv_blocks', 1, 3, default=2)):
        filters = hp.Choice(f'filters_{i}', [32, 64, 128])
        x = Conv2D(filters, (3,3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D()(x)
        x = Dropout(hp.Float(f'dropout_{i}', 0.1, 0.5, step=0.1))(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(hp.Int('dense_units', 64, 256, step=64), activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(hp.Choice('learning_rate', [1e-3, 1e-4, 5e-4])),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_resnet50_model(input_shape):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Freeze first layers
    for layer in base_model.layers[:-30]:  # Unfreeze last few layers
        layer.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=base_model.input, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def load_dataset(folder):
    clean_folder = os.path.join(folder, 'clean')
    stego_folder = os.path.join(folder, 'stego')
    clean_images = [os.path.join(clean_folder, f) for f in os.listdir(clean_folder) if f.lower().endswith('.png')]
    stego_images = [os.path.join(stego_folder, f) for f in os.listdir(stego_folder) if f.lower().endswith('.png')]

    image_paths = clean_images + stego_images
    labels = [0] * len(clean_images) + [1] * len(stego_images)

    return image_paths, labels

def prepare_dataset(image_paths, labels, batch_size=32, training=False):
    def load_image(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, [512, 512])
        img = preprocess_input(img)
        return img, label

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        dataset = dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    else:
        dataset = dataset.batch(batch_size)
    return dataset

def main():
    train_folder = 'archive/train/train'
    test_folder = 'archive/test/test'
    validation_folder = 'archive/val/val'

    train_image_paths, train_labels = load_dataset(train_folder)
    validation_image_paths, validation_labels = load_dataset(validation_folder)
    test_image_paths, test_labels = load_dataset(test_folder)

    train_dataset = prepare_dataset(train_image_paths, train_labels, training=True)
    validation_dataset = prepare_dataset(validation_image_paths, validation_labels)
    test_dataset = prepare_dataset(test_image_paths, test_labels)

    tuner = kt.Hyperband(
        build_model,
        objective='val_accuracy',
        max_epochs=20,
        factor=3,
        directory='my_dir',
        project_name='cnn_steganalysis'
    )

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    best_model_path = 'best_modelCNN.keras'
    model_checkpoint = ModelCheckpoint(best_model_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights = {i: class_weights[i] for i in range(len(class_weights))}

    tuner.search(train_dataset, epochs=20, validation_data=validation_dataset, callbacks=[reduce_lr, early_stopping])

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    model = tuner.hypermodel.build(best_hps)
    history = model.fit(train_dataset, epochs=25, validation_data=validation_dataset, class_weight=class_weights, callbacks=[reduce_lr, early_stopping, model_checkpoint])

    model.save('model/modelCNN.keras')

    y_true = np.array([label for _, label in test_dataset.unbatch().as_numpy_iterator()]).astype(float)
    y_pred = model.predict(test_dataset)
    y_pred_classes = (y_pred > 0.5).astype(int).flatten()

    print("CNN Classification Report:")
    print(classification_report(y_true, y_pred_classes, target_names=['Clean', 'Stego']))

    print("CNN Confusion Matrix:")
    conf_matrix = confusion_matrix(y_true, y_pred_classes)
    print(conf_matrix)

    accuracy = np.mean(y_pred_classes == y_true)
    print("CNN Accuracy:", accuracy)

    precision = precision_score(y_true, y_pred_classes)
    print("CNN Precision:", precision)

    recall = recall_score(y_true, y_pred_classes)
    print("CNN Recall:", recall)

    f1 = f1_score(y_true, y_pred_classes)
    print("CNN F1 Score:", f1)

    tn, fp, fn, tp = conf_matrix.ravel()
    specificity = tn / (tn + fp)
    print("CNN Specificity:", specificity)

    fpr = fp / (fp + tn)
    print("CNN False Positive Rate (FPR):", fpr)

    roc_auc = roc_auc_score(y_true, y_pred)
    print("CNN AUC-ROC:", roc_auc)

if __name__ == "__main__":
    main()