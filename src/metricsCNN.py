import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, recall_score, f1_score, precision_score, roc_auc_score

class ModelTester:
    def __init__(self, model_path, test_folder, batch_size=32, target_size=(512, 512)):
        self.model_path = model_path
        self.test_folder = test_folder
        self.batch_size = batch_size
        self.target_size = target_size
        self.model = self.load_model()
        self.test_generator = self.create_test_generator()

    def load_model(self):
        return load_model(self.model_path)

    def create_test_generator(self):
        test_datagen = ImageDataGenerator(rescale=1.0/255.0)
        return test_datagen.flow_from_directory(
            self.test_folder,
            target_size=self.target_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )

    def evaluate_model(self):
        y_true = self.test_generator.classes
        y_pred = self.model.predict(self.test_generator)
        y_pred_classes = np.argmax(y_pred, axis=1)

        report = classification_report(y_true, y_pred_classes, target_names=self.test_generator.class_indices.keys())
        cm = confusion_matrix(y_true, y_pred_classes)
        recall = recall_score(y_true, y_pred_classes, average='weighted')
        f1 = f1_score(y_true, y_pred_classes, average='weighted')
        precision = precision_score(y_true, y_pred_classes, average='weighted')
        accuracy = np.mean(y_true == y_pred_classes)
        
        # Convert y_true to one-hot encoding for roc_auc_score
        y_true_one_hot = np.zeros((y_true.size, y_pred.shape[1]))
        y_true_one_hot[np.arange(y_true.size), y_true] = 1
        roc_auc = roc_auc_score(y_true_one_hot, y_pred, multi_class='ovr')

        print("Classification Report:\n", report)
        print("Confusion Matrix:\n", cm)
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")
        print(f"Precision: {precision}")
        print(f"Accuracy: {accuracy}")
        print(f"ROC AUC Score: {roc_auc}")

# Usage
if __name__ == "__main__":
    model_path = 'model/best_modelCNN.keras'
#    test_folder = '/home/jovyan/.cache/kagglehub/datasets/marcozuppelli/stegoimagesdataset/versions/2/test/test'
    test_folder = 'archive/test/test'
    tester = ModelTester(model_path, test_folder)
    tester.evaluate_model()