# Tensorflow Custom Dataset by Md Rehan (mdrehan4all)

from tensorflow import keras
import tensorflow as tf

img_height, img_width = (50, 50)
batch_size = 32

train_data_dir = "dataset/train"
valid_data_dir = "dataset/val"
test_data_dir = "dataset/test"

train_generator = tf.keras.preprocessing.image_dataset_from_directory(
    train_data_dir,
    labels="inferred",
    label_mode="int", 
    class_names=['0', '1', '2'],
    color_mode="grayscale",
    batch_size=batch_size,
    image_size=(img_height, img_width),
    shuffle=True,
    seed=43,
    validation_split=0.4,
    subset="training",
)
'''
valid_generator = tf.keras.preprocessing.image_dataset_from_directory(
    valid_data_dir,
    labels="inferred",
    label_mode="int", 
    class_names=['0', '1', '2'],
    color_mode="grayscale",
    batch_size=batch_size,
    image_size=(img_height, img_width),
    shuffle=True,
    seed=43,
    validation_split=0.4,
    subset="validation",
)

test_generator = tf.keras.preprocessing.image_dataset_from_directory(
    train_data_dir,
    labels="inferred",
    label_mode="int",
    class_names=['0', '1', '2'],
    color_mode="grayscale",
    batch_size=1,
    image_size=(img_height, img_width),
    shuffle=True,
    seed=43,
    validation_split=0.4,
    subset="validation",
)
'''
model = keras.Sequential([
  keras.layers.Rescaling(1./255, input_shape=(img_height, img_width, 1)),
  keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
  keras.layers.MaxPooling2D(),
  keras.layers.Flatten(),
  keras.layers.Dense(128, activation='relu'),
  #keras.layers.Dense(64, activation='relu'),
  keras.layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(), metrics = ['accuracy'])
model.fit(train_generator, epochs=10)
model.save('model.h5')