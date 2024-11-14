import numpy as np

class ImagePreprocessor:
    def __init__(self, target_size=(512, 512)):
        self.target_size = target_size

    def preprocess_image(self, image):
        image = image.convert('RGB')
        image = image.resize(self.target_size)
        image = np.array(image).astype(np.float32)
        image /= 255.0  # Normalize to [0, 1]
        return image

    def preprocess_images(self, images):
        return np.array([self.preprocess_image(image) for image in images])