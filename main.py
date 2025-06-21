import streamlit as st
import numpy as np
import cv2
import pandas as pd
from PIL import Image
from segmentation_metod import OtsuSegmentation, OtsuWithFiltersSegmentation, MeanSegmentation
# st.set_page_config(layout="wide")
image = None
with st.sidebar:
    image_path = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    methods = {
    "Otsu": OtsuSegmentation(),
    "Otsu with Filters": OtsuWithFiltersSegmentation(),
    "Mean": MeanSegmentation()
    }
    selected = st.multiselect("Metody segmentacji", methods)

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
            st.subheader(f"Segmentacja metodą: {method}")
            segmented_image = method.segment(image)
            st.image(segmented_image, caption=f"Segmentacja przy pomocy {method}", use_column_width=True)
    else:
        st.error("Please upload an image to segment.")