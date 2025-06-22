import streamlit as st
import numpy as np
import cv2
import pandas as pd
from PIL import Image
from segmentation_metod import OtsuSegmentation, OtsuWithFiltersSegmentation, MeanSegmentation, UNetSegmentation
from unet_model import load_model
# st.set_page_config(layout="wide")
image = None
with st.sidebar:
    image_path = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    methods = {
    "Otsu": OtsuSegmentation(),
    "Otsu with Filters": OtsuWithFiltersSegmentation(),
    "Mean": MeanSegmentation(),
    "U-Net": UNetSegmentation(load_model("unet_model.h5"))
    }
    selected = st.multiselect("Select segmentation methods", methods)

    if image_path is not None:
        # Read the image
        image = Image.open(image_path)
        if image is None:
            st.error("Error reading the image.")
        else:
            st.image(image, caption="Uploaded Image", use_column_width=True)


if selected:
    if image:
        image = np.array(image)
        for method_name in selected:
            method = methods[method_name]
            st.subheader(f"Segmentation using {method_name.lower()}")
            segmented_image = method.segment(image)
            st.image(segmented_image, caption=method, use_column_width=True)