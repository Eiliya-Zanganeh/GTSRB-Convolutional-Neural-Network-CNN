## Traffic Sign Classification using the GTSRB Dataset with TensorFlow
This project trains a Convolutional Neural Network (CNN) to classify images of traffic signs using the German Traffic Sign Recognition Benchmark (GTSRB) dataset.

## Prerequisites
Ensure the following dependencies are installed:

* Python 3.x
* TensorFlow
* OpenCV
* NumPy
* scikit-learn
You can install the required libraries by running:
```bash
pip insall tensorflow opencv-python numpy scikit-learn
```

## Dataset
The GTSRB dataset contains 43 different classes of traffic signs. Each class is represented by a folder in the dataset, where each folder contains the corresponding images. Make sure the dataset is structured in subfolders within a gtsrb directory, with the folder names corresponding to the traffic sign categories (0-42).

## Code Overview
1. Loading Data
The load_data function iterates through the dataset directory and loads the images, resizing each image to 30x30 pixels, and stores them in the datas list. The labels are derived from the folder names (0-42) and stored in the labels list.

2. Preprocessing and Train-Test Split
The images and labels are split into training and testing sets using train_test_split from sklearn. The labels are one-hot encoded using tf.keras.utils.to_categorical. The pixel values of the images are normalized to the range [0, 1].

3. Model Architecture
The CNN model is built using tf.keras.Sequential and contains the following layers:

* Conv2D: 32 filters of size 3x3 followed by ReLU activation
* MaxPooling2D: Pooling layer with size 2x2
* Conv2D: Another 32 filters of size 3x3 followed by ReLU activation
* MaxPooling2D: Pooling layer with size 2x2
* Flatten: Flattens the 2D data into a 1D vector
* Dense: Fully connected layer with 128 units and ReLU activation
* Dropout: Regularization with a dropout rate of 0.5
* Dense: Output layer with 43 units (one for each category) and softmax activation

4. Training the Model
The model is trained for 10 epochs on the training data using model.fit.

5. Evaluating the Model
After training, the model is evaluated on the test dataset using model.evaluate, which provides the accuracy of the model on unseen data.

6. Saving the Model
If a filename is provided as a command-line argument, the trained model is saved to that file using model.save.

How to Run
Prepare the dataset: Ensure that the GTSRB dataset is downloaded and placed in the gtsrb directory.
Run the script: Execute the script by running the following command:
```bash
python traffic/traffic.py
```
## Customization
* Dataset: The load_data function can be modified to load data from a different directory or to process images differently.
* Model Architecture: You can modify the get_model function to adjust the architecture of the CNN (e.g., by adding more layers or changing the number of filters).
* Hyperparameters: Change the number of epochs, batch size, learning rate, or optimizer according to your needs.
