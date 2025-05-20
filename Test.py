from ultralytics import YOLO

# Importation du modèle déja entrainé
modelc = YOLO("runs/detect/train/weights/best.pt")


# test sur une image
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np # Import numpy for array conversion

im1 = Image.open("PAP-DetectionPanneauSignalisation/datasets/coco/images/train/Capture d'écran 2025-03-21 225646.jpg")
results = modelc.predict(source=im1.convert("RGB"), save=True)

# Access the image data from the prediction results
image_with_predictions = results[0].plot()  # Assuming the image is the first element in the results list


image_with_predictions = cv2.cvtColor(image_with_predictions, cv2.COLOR_BGR2RGB)

# Display the original image and the image with predictions side by side
plt.figure(figsize=(12, 6))

# Original image
plt.subplot(1, 2, 1)
plt.imshow(im1)
plt.title("Original Image")
plt.axis('off')

# Image with predictions
plt.subplot(1, 2, 2)
plt.imshow(image_with_predictions)
plt.title("Image with Predictions")
plt.axis('off')

plt.tight_layout()
plt.show()