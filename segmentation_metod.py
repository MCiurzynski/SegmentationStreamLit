import cv2
import numpy as np
import tensorflow as tf
import streamlit as st

class SegmentationMethod:
    def __init__(self, name: str):
        self.name = name

    def segment(self, data):
        raise NotImplementedError("This method should be overridden by subclasses")
    
    def show(self, data):
        st.subheader(f"Segmentation using {self.name.lower()}")
        segmented_data = self.segment(data)
        st.image(segmented_data, caption=self.name, use_column_width=True)

    def __str__(self):
        return f"{self.name}"

    def __repr__(self):
        return self.__str__()


class ManualThresholdSegmentation(SegmentationMethod):
    def __init__(self):
        super().__init__("Manual Threshold")

    def show(self, data):
        st.subheader(f"Segmentation using {self.name.lower()}")
        self.threshold = st.slider("Threshold value", 0, 255, 127)
        
        segmented_data = self.segment(data)
        st.image(segmented_data, caption=self.name, use_column_width=True)

    def segment(self, data):
        if len(data.shape) == 3:  # Check if the image is colored
            gray = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
        else:
            gray = data

        _, segmented_image = cv2.threshold(gray, self.threshold, 255, cv2.THRESH_BINARY_INV)
        return segmented_image

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

    def show(self, data):
        st.subheader(f"Segmentation using {self.name.lower()}")

        self.sigma = st.slider("Sigma for Gaussian Blur", 0.0, 10.0, 1.0, key="sigma_blur_otsu")
        self.size = st.slider("Size for Morphological Operations", 0, 100, 40, key="size_morph_otsu")
        
        segmented_data = self.segment(data)
        st.image(segmented_data, caption=self.name, use_column_width=True)

    def segment(self, data):
        if len(data.shape) == 3:  # Check if the image is colored
            gray = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
        else:
            gray = data

        # Apply Gaussian blur to reduce noise
        if self.sigma != 0:
            gray = cv2.GaussianBlur(gray, (0, 0), sigmaX=self.sigma, sigmaY=self.sigma)
        _, segmented_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        
        if self.size != 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.size, self.size))
            segmented_image = cv2.morphologyEx(segmented_image, cv2.MORPH_OPEN, kernel)
            segmented_image = cv2.morphologyEx(segmented_image, cv2.MORPH_CLOSE, kernel)
        return segmented_image

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

class UNetWithFiltersSegmentation(UNetSegmentation):
    def __init__(self, model):
        super().__init__(model)

    def show(self, data):
        st.subheader(f"Segmentation using {self.name.lower()}")
        col1, col2 = st.columns([1, 3])
        with col1:
            self.sigma = st.slider("Sigma for Gaussian Blur", 0.0, 10.0, 1.0, key="sigma_blur_unet")
            self.size = st.slider("Size for Morphological Operations", 0, 100, 40, key="size_morph_unet")
        
        segmented_data = self.segment(data)
        with col2:
            st.image(segmented_data, caption=self.name, use_column_width=True) 

    def segment(self, data):
        if self.sigma != 0:
            data = cv2.GaussianBlur(data, (0, 0), sigmaX=self.sigma, sigmaY=self.sigma)
        prediction = super().segment(data)
        
        if self.size != 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.size, self.size))
            prediction = cv2.morphologyEx(prediction, cv2.MORPH_OPEN, kernel)
            prediction = cv2.morphologyEx(prediction, cv2.MORPH_CLOSE, kernel)
        
        return prediction