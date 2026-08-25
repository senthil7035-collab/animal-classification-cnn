from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load trained model
model = load_model("animal_classifier.h5")

# Class names from Animals-10 dataset
classes = [
    "dog",        # cane
    "horse",      # cavallo
    "elephant",   # elefante
    "butterfly",  # farfalla
    "chicken",    # gallina
    "cat",        # gatto
    "cow",        # mucca
    "sheep",      # pecora
    "spider",     # ragno
    "squirrel"    # scoiattolo
]

# Load test image
img = image.load_img("test1.jpg", target_size=(128, 128))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img = img / 255.0

# Predict
prediction = model.predict(img)

predicted_class = classes[np.argmax(prediction)]
confidence = np.max(prediction) * 100

print("\nPredicted Animal:", predicted_class)
print("Confidence:", round(confidence, 2), "%")
