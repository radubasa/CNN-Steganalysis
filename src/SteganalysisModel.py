from tensorflow.keras.models import load_model

class SteganalysisModel:
    def __init__(self, model_path):
        self.model = load_model(model_path)

    def predict(self, images):
        predictions = self.model.predict(images)
        return np.argmax(predictions, axis=1)