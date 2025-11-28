import numpy as np
from PIL import Image, ImageOps
def preprocess_image(image, target_size=(150, 150)):
    image = ImageOps.fit(image, target_size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image) / 255.0
    img_reshape = img_array[np.newaxis, ...]
    return img_reshape
