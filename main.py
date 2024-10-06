from convert_xml import xml_df
import pandas as pd
from PIL import Image
import os
xml_df = pd.read_csv('pothole_labels.csv')
import numpy as np
from sklearn.model_selection import train_test_split

# Split dataset into train, validation, and test sets
train_df, test_df = train_test_split(xml_df, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)

import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def load_image_and_label(image_path, label):
    image = load_img(image_path, target_size=(128, 128))  # resize the image
    image = img_to_array(image) / 255.0  # normalize the image
    return image, label


def create_dataset(image_dir, dataset):
    data = []

    for index, row in dataset.iterrows():
        # Use 'filename' instead of 'image_id'
        image_path = os.path.join(image_dir, row['filename'])

        # Extract bounding box info
        bbox = {
            'class': row['class'],
            'xmin': row['xmin'],
            'ymin': row['ymin'],
            'xmax': row['xmax'],
            'ymax': row['ymax']
        }

        # Assuming you're saving this data for further use
        data.append({
            'image_path': image_path,
            'bbox': bbox
        })

    return data

# Create datasets for train, validation, and test
train_dataset = create_dataset(train_df, '/Users/sangi/Documents/pothole_proj/data/annotated-images')
val_dataset = create_dataset(val_df, '/Users/sangi/Documents/pothole_proj/data/annotated-images')
test_dataset = create_dataset(test_df, '/Users/sangi/Documents/pothole_proj/data/annotated-images')

# Batching and shuffling the datasets
train_dataset = train_dataset.batch(32).shuffle(1000)
val_dataset = val_dataset.batch(32)
test_dataset = test_dataset.batch(32)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential()

# Convolutional layers
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the output
model.add(Flatten())

# Fully connected layers
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4))  # Output for bounding box coordinates: xmin, ymin, xmax, ymax

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

model.summary()

model.fit(train_dataset.batch(32), validation_data=val_dataset.batch(32), epochs=10)

test_loss, test_accuracy = model.evaluate(test_dataset.batch(32))
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')

predictions = model.predict(test_dataset.batch(32))
print(predictions)
