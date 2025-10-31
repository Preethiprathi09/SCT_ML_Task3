import streamlit as st
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from PIL import Image

st.title("🐾 Cat vs Dog Classifier - SkillCraft Task 3")

# ----------------------
# Dataset path
IMG_SIZE = 64
data_dir = r"C:\Users\Admin\Desktop\SkillCraft\Task3\training_set\training_set"

@st.cache_data
def load_data(data_dir):
    data, labels = [], []
    for category in ["cats","dogs"]:
        folder = os.path.join(data_dir, category)
        label = 0 if category=="cats" else 1
        for img_name in os.listdir(folder):
            img_path = os.path.join(folder, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                data.append(img.flatten())
                labels.append(label)
    return np.array(data), np.array(labels)

# ----------------------
# Load dataset
st.subheader("📁 Loading Dataset")
X, y = load_data(data_dir)
st.success(f"Dataset loaded! Total images: {len(X)}")

# ----------------------
# Split & Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

st.subheader("⚙️ Train SVM Model")
kernel_option = st.selectbox("Choose SVM Kernel:", ["linear","rbf","poly"])

if st.button("Train Model"):
    st.info("Training SVM model... ⏳")
    model = SVC(kernel=kernel_option)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.success(f"✅ Model trained! Accuracy: {acc*100:.2f}%")
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred, target_names=["Cat","Dog"]))
    st.session_state.model = model
else:
    st.warning("Click 'Train Model' to start training.")

# ----------------------
# Upload image for prediction
st.subheader("🐕 Predict an Image")
uploaded_file = st.file_uploader("Upload a cat/dog image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=200)
    img_array = np.array(image)
    img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE)).flatten().reshape(1,-1)
    if "model" in st.session_state:
        pred = st.session_state.model.predict(img_array)[0]
        label = "🐱 Cat" if pred==0 else "🐶 Dog"
        st.success(f"Prediction: {label}")
    else:
        st.warning("Train the model first!")
