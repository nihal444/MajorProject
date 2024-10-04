import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2

# from google.colab import drive

# drive.mount('/content/drive')

# Define the F1Score metric
class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * ((p * r) / (p + r + keras.backend.epsilon()))

    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()

    def get_config(self):
        base_config = super().get_config()
        return base_config

# Set up paths and parameters
model_path = './'
IMG_SIZE = (224, 224)
class_names = ['MEL', 'NV', 'BCC', 'SCC']

# Load the model with custom objects
try:
    model_saved = tf.keras.models.load_model(model_path, custom_objects={'F1Score': F1Score})
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please check if the model file exists and the path is correct.")
    exit()

def predict_image(img, model, class_names, nv_threshold=0.7):
    img_array = cv2.resize(img, IMG_SIZE)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32) / 255.0

    prediction = model.predict(img_array)

    if np.argmax(prediction) == class_names.index('NV') and prediction[0][class_names.index('NV')] < nv_threshold:
        predicted_class = class_names[np.argsort(prediction[0])[-2]]
    else:
        predicted_class = class_names[np.argmax(prediction)]

    confidence = np.max(prediction)

    return predicted_class, confidence

# Initialize the webcam
cap = cv2.VideoCapture(0)

