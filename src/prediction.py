import numpy as np
import tensorflow as tf

def make_prediction(model, preprocessed_image):
    # Run inference
    prediction = model.predict(preprocessed_image)
    
    # Map index to class label
    classes = ['Paper', 'Rock', 'Scissors']
    class_idx = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    
    return classes[class_idx], confidence
