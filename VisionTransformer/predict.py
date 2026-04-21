"""
WHY THIS FILE?
Used to test one image after training with Vision Transformer.

You give image path ->
Model predicts fault class
"""

import numpy as np
import tensorflow as tf
from keras.preprocessing import image
from config import *
import keras
keras.config.enable_unsafe_deserialization()
# Load trained model
from vit_keras.layers import ClassToken, AddPositionEmbs
model = tf.keras.models.load_model(
	MODEL_PATH,
	custom_objects={
		'ClassToken': ClassToken,
		'AddPositionEmbs': AddPositionEmbs
	}
)

# -------------------------------------------
# Enter image path here
# -------------------------------------------
img_path = "/Users/punithkumarr/Desktop/SPFD/faultsdataset/test/dusty/Dust (10).jpg"  # Change this to your test image

# Load image
img = image.load_img(img_path, target_size=IMG_SIZE)
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)
class_index = np.argmax(prediction)
confidence = np.max(prediction)

print("Predicted Class:", CLASS_NAMES[class_index])
print("Confidence:", round(float(confidence)*100, 2), "%")
