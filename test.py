# Tensorflow Custom Dataset by Md Rehan (mdrehan4all)

from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

img_height, img_width = (50, 50)

# Load the model
model = load_model('model.h5')

data = np.ndarray(shape=(1, img_height, img_width), dtype=np.float32)
image = Image.open('image1.jpg')
image = image.convert('L')
image = ImageOps.fit(image, (img_height, img_width), Image.ANTIALIAS)
image_array = np.asarray(image)
# normalized_image_array = (image_array.astype(np.float32) / 255.0) - 1
data[0] = image_array # normalized_image_array

# run the inference
prediction = model.predict(data)
print(prediction)