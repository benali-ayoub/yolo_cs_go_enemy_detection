import numpy as np
from tensorflow import keras
import cv2


def load_model(model_path):
    """
    Loads a Keras model from a specified path.
    
    Parameters:
    - model_path (str): Path to the saved Keras model (.h5 file)
    
    Returns:
    - model (keras.Model): Loaded Keras model
    """
    try:
        model = keras.models.load_model(model_path)
        print(f"Model loaded successfully from: {model_path}")
        return model
    except OSError as e:
        print(f"Failed to load model from {model_path}: {e}")   


def predict_image(image_path, model):
    image = cv2.imread(image_path)
    original_height, original_width = image.shape[:2]
    
    # Preprocess image
    resized_image = cv2.resize(image, (224, 224)) / 255.0
    input_data = np.expand_dims(resized_image, axis=0)

    # Model prediction
    bbox_preds, class_preds = model.predict(input_data)
    
    # Rescale bounding box to original image size
    x_min, y_min, x_max, y_max = bbox_preds[0]
    x_min = int(x_min * original_width)
    y_min = int(y_min * original_height)
    x_max = int(x_max * original_width)
    y_max = int(y_max * original_height)
    
    # Display results
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
    cv2.imshow("Prediction", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

model = load_model('/models/faster_rcnn_model.keras')
predict_image('1.jpg', model)
