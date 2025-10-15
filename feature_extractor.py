import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
input_size = (224, 224)

def extract_features(img: Image.Image):
    img = img.convert('RGB').resize(input_size)
    x = np.expand_dims(np.array(img), axis=0)
    x = preprocess_input(x)
    features = model.predict(x, verbose=0)[0]
    features = features / np.linalg.norm(features)
    return features
