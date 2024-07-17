# Building up the application for using the model created and identifying digits  
import tensorflow as tf  
from PIL import Image as im  
import numpy as np  

# Importing the model created  
model = tf.keras.models.load_model("Models/digit_recog_cnn_model.keras")

# Defining function to convert image to grayscale and resize  
def preprocess_image(image_path):  
    img = im.open(image_path).convert('L')  # Convert image to grayscale  
    img_resized = img.resize((28, 28), im.LANCZOS)  # Resize it to 28x28
    image_array = np.array(img_resized).reshape(1, 28, 28, 1) / 255.0  # Reshape and normalize the image  
    return image_array  

def main():  
    print("This is a Handwritten Digit Recognizing System")  
    image_path = input("Enter the image path: ")  
    img = preprocess_image(image_path)  
    
    if img is not None:  
        digit = model(img)  
        print("The predicted digit is:", np.argmax(digit))  

main()