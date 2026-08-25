import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# DATASET PATH
# =====================================================

dataset_path = "raw-img"


# =====================================================
# IMAGE PREPROCESSING
# =====================================================

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# =====================================================
# CLASS NAMES
# =====================================================

print("\nClass Labels:")
print(train_data.class_indices)

# =====================================================
# CNN MODEL
# =====================================================

model = Sequential()

model.add(
    Conv2D(
        32,
        (3, 3),
        activation='relu',
        input_shape=(128, 128, 3)
    )
)
model.add(MaxPooling2D(2, 2))

model.add(
    Conv2D(
        64,
        (3, 3),
        activation='relu'
    )
)
model.add(MaxPooling2D(2, 2))

model.add(
    Conv2D(
        128,
        (3, 3),
        activation='relu'
    )
)
model.add(MaxPooling2D(2, 2))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))

# =====================================================
# COMPILE MODEL
# =====================================================

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# =====================================================
# MODEL SUMMARY
# =====================================================

model.summary()

# =====================================================
# TRAIN MODEL
# =====================================================

history = model.fit(
    train_data,
    validation_data=val_data,
   epochs=20
)

# =====================================================
# SAVE MODEL
# =====================================================

model.save("animal_classifier.h5")

print("\nModel Saved Successfully!")

# =====================================================
# PLOT ACCURACY
# =====================================================

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")

plt.legend(
    ['Train Accuracy', 'Validation Accuracy'],
    loc='lower right'
)

plt.show()
