# Import the tensorflow
import tensorflow as tf  

# Load the digits dataset  
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()  

# Normalize the data by dividing it by 255  
x_train, x_test = x_train / 255.0, x_test / 255.0  

# Reshape data to have a single channel  
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)  
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)  

# Define the CNN model  
model = tf.keras.Sequential([  
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),  
    tf.keras.layers.MaxPooling2D((2, 2)),  
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),  
    tf.keras.layers.MaxPooling2D((2, 2)),  
    tf.keras.layers.Flatten(),  
    tf.keras.layers.Dense(128, activation='relu'),  
    tf.keras.layers.Dropout(0.2),  
    tf.keras.layers.Dense(10)  
])  

# Compile the model  
model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])  

# Train the model  
model.fit(x_train, y_train, epochs=20)  

# Evaluate the model on the test set  
test_loss, test_acc = model.evaluate(x_test, y_test)  
print("Test accuracy:", test_acc)  

# Save the model to use it in the application (run.py)
model.save("digit_recog_cnn_model.keras")