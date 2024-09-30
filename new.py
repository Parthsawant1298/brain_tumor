from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.preprocessing import image
import os
from PIL import Image
import base64
import io

app = Flask(__name__)

# Define paths to the datasets
train_dir = r"C:\Users\parth sawant\Desktop\brain_tumor final\MRI Image Dataset for Brain Tumor\Training"
test_dir = r'C:\Users\parth sawant\Desktop\brain_tumor final\MRI Image Dataset for Brain Tumor\Testing'

# Preprocessing the Training set
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

training_set = train_datagen.flow_from_directory(train_dir,
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='categorical')

# Preprocessing the Test set
test_datagen = ImageDataGenerator(rescale=1./255)

test_set = test_datagen.flow_from_directory(test_dir,
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='categorical')

# Part 2 - Building the CNN

# Initialising the CNN
cnn = tf.keras.models.Sequential()

# Step 1 - Convolution
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))

# Step 2 - Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Adding a second convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Step 3 - Flattening
cnn.add(tf.keras.layers.Flatten())

# Step 4 - Full Connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# Step 5 - Output Layer
cnn.add(tf.keras.layers.Dense(units=4, activation='softmax'))

# Part 3 - Training the CNN

# Compiling the CNN
cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Training the CNN on the Training set and evaluating it on the Test set
history = cnn.fit(x=training_set,
                  validation_data=test_set,
                  epochs=25,
                  verbose=1)

# Save the model
model_path = r'C:\Users\parth sawant\Desktop\brain_tumor final\mri.keras'
cnn.save(model_path)

# Load the model
cnn = tf.keras.models.load_model(model_path)

@app.route('/upload', methods=['GET', 'POST'])
def upload_and_predict():
    if request.method == 'POST':
        # Define your class labels based on your dataset
        class_labels = ['glioma', 'meningioma', 'no tumor', 'pituitary']

        # Print class labels and their corresponding indices
        class_indices = {label: index for index, label in enumerate(class_labels)}
        print(f'Class indices: {class_indices}')

        # Process the uploaded image
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            # Convert the file to an image
            img = Image.open(uploaded_file.stream)
            img = img.resize((64, 64))  # Resize to match training images
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0  # Ensure scaling matches training preprocessing

            # Predict the class
            result = cnn.predict(img_array)

            # Get the predicted class index
            predicted_class_index = np.argmax(result[0])
            prediction = class_labels[predicted_class_index]

            # Print the result for debugging
            print(f'Prediction probabilities: {result[0]}')
            print(f'Predicted class index: {predicted_class_index}')
            print(f'Predicted class label: {prediction}')

            return render_template('result.html', prediction=prediction)
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
