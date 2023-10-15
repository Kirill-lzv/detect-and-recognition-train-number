import tensorflow as tf
import json
import cv2
import numpy as np
import os

image_dir = "/content/drive/MyDrive/ai_13.10.23/images_full"
json_dir = "/content/drive/MyDrive/ai_13.10.23/labels_full"

all_images = []
all_coordinates = []

desired_height, desired_width = 600, 600

for json_file in os.listdir(json_dir):
    if json_file.endswith(".json"):
        with open(os.path.join(json_dir, json_file), "r") as file:
            data = json.load(file)

        for item in data:
            image_path = os.path.join(image_dir, item["image"])
            image = cv2.imread(image_path)
            if image is not None:
                image = cv2.resize(image, (desired_width, desired_height))
                all_images.append(image / 255.0)

                coordinates = item["annotations"][0]["coordinates"]
                x = coordinates["x"]
                y = coordinates["y"]
                width = coordinates["width"]
                height = coordinates["height"]

                all_coordinates.append([x, y, width, height])
                # print(1)

all_images = np.array(all_images, dtype=np.float32)
all_coordinates = np.array(all_coordinates, dtype=np.float32)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(desired_height, desired_width, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(4)
])

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(all_images, all_coordinates, epochs=212)

model.save("/content/drive/MyDrive/ai_13.10.23/mod_212_600.h5")
