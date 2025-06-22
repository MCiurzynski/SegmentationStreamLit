import cv2
import numpy as np
import tensorflow as tf
import streamlit as st

class SegmentationMethod:
    def __init__(self, name: str):
        self.name = name

    def segment(self, data):
        raise NotImplementedError("This method should be overridden by subclasses")

    def __str__(self):
        return f"{self.name}"

    def __repr__(self):
        return self.__str__()

class OtsuSegmentation(SegmentationMethod):
    def __init__(self):
        super().__init__("Otsu")

    def segment(self, data):
        if len(data.shape) == 3:  # Check if the image is colored
            gray = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
        else:
            gray = data

        _, segmented_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return segmented_image

class OtsuWithFiltersSegmentation(SegmentationMethod):
    def __init__(self):
        super().__init__("Otsu with Filters")

    def segment(self, data):
        if len(data.shape) == 3:  # Check if the image is colored
            gray = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
        else:
            gray = data

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (1, 1), 0)
        _, segmented_image = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        opened_image = cv2.morphologyEx(segmented_image, cv2.MORPH_OPEN, np.ones((40, 40), np.uint8))
        closed_image = cv2.morphologyEx(opened_image, cv2.MORPH_CLOSE, np.ones((40, 40), np.uint8))
        return closed_image

class MeanSegmentation(SegmentationMethod):
    def __init__(self):
        super().__init__("Mean")

    def segment(self, data):
        if len(data.shape) == 3:  # Check if the image is colored
            gray = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
        else:
            gray = data

        mean_value = np.mean(gray)
        _, segmented_image = cv2.threshold(gray, mean_value, 255, cv2.THRESH_BINARY_INV)
        return segmented_image

class UNetSegmentation(SegmentationMethod):
    def __init__(self, model):
        super().__init__("U-Net")
        self.model = model

    def segment(self, data):
        original_size = data.shape[:2]
        data = cv2.resize(data, (224, 224))  # Resize to match model input
        data = np.expand_dims(data, axis=0)  # Add batch dimension
        data = data.astype(np.float32) / 255.0

        prediction = self.model.predict(data)
        prediction = np.squeeze(prediction)  # Remove batch dimension
        prediction = (prediction > 0.5).astype(np.uint8) * 255  # Binarize the output
        prediction = cv2.resize(prediction.squeeze(), (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)
        return prediction
    