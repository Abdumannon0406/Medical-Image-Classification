from ultralytics import YOLO

# Load the trained model
model=YOLO("patth/to/best")

# Testing the model
results=model.predict("path/to/test_image",save=True)