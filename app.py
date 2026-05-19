import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from processing import (
    apply_canny,
    apply_sobel,
    apply_laplacian
)

from metrics import edge_pixel_count


st.set_page_config(layout="wide")

st.title("Interactive Edge Detection Explorer")

st.write(
    """
Explore how different edge detection algorithms behave under
different parameter settings.
"""
)

# ---------------- Sidebar ----------------

st.sidebar.header("Controls")

uploaded_file = st.sidebar.file_uploader(
    "Upload an image",
    type=["png", "jpg", "jpeg"]
)

method = st.sidebar.selectbox(
    "Edge Detection Method",
    ["Canny", "Sobel", "Laplacian"]
)

threshold1 = st.sidebar.slider(
    "Lower Threshold",
    0,
    255,
    50
)

threshold2 = st.sidebar.slider(
    "Upper Threshold",
    0,
    255,
    150
)

# ---------------- Load image ----------------

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

else:
    image = Image.open("sample_images/cat.jpg").convert("RGB")

image_np = np.array(image)

# ---------------- Processing ----------------

if method == "Canny":
    result = apply_canny(image_np, threshold1, threshold2)

elif method == "Sobel":
    result = apply_sobel(image_np)

elif method == "Laplacian":
    result = apply_laplacian(image_np)

# ---------------- Display ----------------

col1, col2 = st.columns(2)

with col1:
    st.subheader("Original Image")
    st.image(image_np)

with col2:
    st.subheader("Edge Detection Result")
    st.image(result)

# ---------------- Metrics ----------------

st.subheader("Metrics")

count = edge_pixel_count(result)

st.metric("Detected Edge Pixels", int(count))

# ---------------- Histogram ----------------

st.subheader("Edge Intensity Histogram")

fig, ax = plt.subplots()

ax.hist(result.ravel(), bins=50)

ax.set_xlabel("Intensity")
ax.set_ylabel("Frequency")

st.pyplot(fig)

# ---------------- Interpretation ----------------

st.subheader("Interpretation")

st.write(
    """
- Lower thresholds detect more edges but can increase noise.
- Higher thresholds produce cleaner edges but may miss fine details.
- Sobel emphasizes gradients.
- Laplacian highlights rapid intensity changes.
- Canny is generally more robust to noise.
"""
)