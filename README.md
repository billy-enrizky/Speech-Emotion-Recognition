# Deep Learning: Speech Emotion Recognition Project

## Importing Necessary Libraries
```python
# Importing necessary libraries
import os  # Operating system functionalities
import librosa  # Audio processing library
import wave  # Module for reading and writing WAV files
import numpy as np  # Numerical operations library
import pandas as pd  # Data manipulation library
import matplotlib.pyplot as plt  # Plotting library

# Importing components for Dividing into the Training Set and the Testing Set
from sklearn.model_selection import train_test_split  # Splitting the dataset for training and testing

# Importing components for Long Short-Term Memory (LSTM) Classifier
import keras  # High-level neural networks API
from tensorflow.keras.utils import to_categorical  # Utility for one-hot encoding
from keras.models import Sequential  # Sequential model for stacking layers
from keras.layers import *  # Different layers for building neural networks
from keras.optimizers import rmsprop  # Optimizer for training the model
```

## Ravdess Emotional Speech Audio
In the development of my Real-Time Speech Emotion Recognition project, I have chosen to leverage the "ravdess-emotional-speech-audio" dataset due to its richness and suitability for training emotion recognition models.

### Ryerson Audio-Visual Database of Emotional Speech and Song (ravdess)
The "ravdess-emotional-speech-audio" dataset, a creation of Ryerson University, consists of 1440 audio files, each lasting approximately 3-5 seconds.

### Diverse Emotional States
This dataset features a diverse set of emotional states, including neutral, calm, happy, sad, angry, fearful, disgust, and surprised. The involvement of professional actors ensures controlled and standardized representations.

### Actor Diversity
With 24 actors split evenly between genders, this dataset contributes to the richness of vocal characteristics, accents, and expressive styles, critical for building a robust Speech Emotion Recognition (SER) model.

### Audio Characteristics
Recorded at 48 kHz and saved in WAV format, the high-quality audio maintains standards suitable for training deep neural networks. A corresponding CSV file provides metadata such as emotion labels, actor information, file paths, and names.

### Focus on Speech Segments
Given the project's focus on real-time speech emotion recognition, attention is concentrated on the speech segments. This approach aligns closely with real-world applications like virtual assistants, customer service, and mental health support.

## Extracting Mel-frequency cepstral coefficients
```python
def extract_mfcc(wav_file_name):
    '''This function retrieves the mean of MFCC features from an input WAV file located 
    at the specified path. The input is the path to the WAV file, and the output is 
    the resulting MFCC features.'''
    
    # Loading the WAV file using librosa and obtaining the audio signal (y) and sampling rate (sr)
    y, sr = librosa.load(wav_file_name)
    
    # Extracting MFCC features with a total of 40 coefficients, and computing the mean across dimensions
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    
    # Returning the resulting MFCC features
    return mfccs
```
Lists are used to store labels and extracted MFCC features for the Ravdess emotional speech dataset.

```python
ravdess_speech_labels = []  
ravdess_speech_data = []

# Iterating through the files in the specified directory
for dirname, _, filenames in os.walk('./ravdess-emotional-speech-audio/'):
    for filename in filenames:
        # Extracting emotion label from the filename and converting to an integer
        ravdess_speech_labels.append(int(filename[7:8]) - 1)
        
        # Obtaining the full path of the WAV file
        wav_file_name = os.path.join(dirname, filename)
        
        # Extracting MFCC features from the WAV file using the previously defined function
        ravdess_speech_data.append(extract_mfcc(wav_file_name))
```

## Converting Data and Labels to Categorical Arrays
```python
# Converting the list of MFCC features into a NumPy array
ravdess_speech_data_array = np.asarray(ravdess_speech_data)

# Converting the list of emotion labels into a NumPy array
ravdess_speech_label_array = np.array(ravdess_speech_labels)

# Converting the integer labels into categorical format using one-hot encoding
labels_categorical = to_categorical(ravdess_speech_label_array)

# Displaying the shapes of the MFCC data array and the categorical label array
ravdess_speech_data_array.shape, labels_categorical.shape
# Output: ((2880, 40), (2880, 8))
```

## Dividing the dataset into training, validation, and testing subsets.
```python
# Splitting the dataset into training and testing sets using train_test_split
x_train, x_test, y_train, y_test = train_test_split(np.array(ravdess_speech_data_array),
                                                    labels_categorical, test_size=0.2,
                                                    random_state=9)

# Calculating the total number of samples in the dataset
number_of_samples = ravdess_speech_data_array.shape[0]

# Determining the number of samples for training, validation, and testing sets
training_samples = int(number_of_samples * 0.

8)
validation_samples = int(number_of_samples * 0.1)
test_samples = int(number_of_samples * 0.1)
```

## Defining the LSTM Model
```python
# Function to create an LSTM model for Speech Emotion Recognition
def create_model_LSTM():
    # Initializing a sequential model
    model = Sequential()
    
    # Adding an LSTM layer with 128 units, not returning sequences, and input shape of (40, 1)
    model.add(LSTM(128, return_sequences=False, input_shape=(40, 1)))
    
    # Adding a Dense layer with 64 units
    model.add(Dense(64))
    
    # Adding a Dropout layer with a dropout rate of 40%
    model.add(Dropout(0.4))
    
    # Adding an Activation layer with ReLU activation function
    model.add(Activation('relu'))
    
    # Adding another Dense layer with 32 units
    model.add(Dense(32))
    
    # Adding a Dropout layer with a dropout rate of 40%
    model.add(Dropout(0.4))
    
    # Adding an Activation layer with ReLU activation function
    model.add(Activation('relu'))
    
    # Adding another Dense layer with 8 units
    model.add(Dense(8))
    
    # Adding an Activation layer with softmax activation function for multiclass classification
    model.add(Activation('softmax'))
    
    # Compiling the model with categorical crossentropy loss, Adam optimizer, and accuracy metric
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    
    # Displaying the model summary
    model.summary()
    
    # Returning the compiled model
    return model
```

## Training the Deep Learning LSTM Model
```python
LSTM_model = create_model_LSTM()

# Training the model with a specific number of epochs
LSTM_model_history = LSTM_model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=121, shuffle=True)
```

## Visualizing the Training Loss and Accuracy
```python
# Function to plot training and validation metrics
def plot_metric(model_training_history, metric_name_1, metric_name_2, plot_name):
    '''
    This function is designed to create a graph displaying the provided metrics.
    Parameters:
        model_training_history: A history object containing recorded training and validation 
                                loss values and metric values across consecutive epochs.
        metric_name_1:          The name of the first metric to be visualized in the graph.
        metric_name_2:          The name of the second metric to be visualized in the graph.
        plot_name:              The title of the graph.
    '''
    # Extract metric values from the training history.
    metric_value_1 = model_training_history.history[metric_name_1]
    metric_value_2 = model_training_history.history[metric_name_2]
    
    # Generate a range of epochs for x-axis.
    epochs = range(len(metric_value_1))
    
    # Plot the first metric in blue.
    plt.plot(epochs, metric_value_1, 'blue', label=metric_name_1)
    
    # Plot the second metric in red.
    plt.plot(epochs, metric_value_2, 'red', label=metric_name_2)
    
    # Set the title of the graph.
    plt.title(str(plot_name))
    
    # Add a legend to the graph.
    plt.legend()

# Plot the training and validation loss metrics for visualization.
plot_metric(LSTM_model_history, 'loss', 'val_loss', 'Total Loss vs Total Validation Loss')

# Plot the training and validation loss metrics for visualization.
plot_metric(LSTM_model_history, 'accuracy', 'val_accuracy', 'Total Accuracy vs Total Validation Accuracy')
```

## Assessing the Trained LSTM Model
```python
# Evaluating the trained model on a separate subset
model_evaluation_history = LSTM_model.evaluate(x_test, y_test)
print(model_evaluation_history)
# Output: [0.0434, 0.9861]
```

## Saving the LSTM Model
```python
import datetime as dt

# Retrieve loss and accuracy from the model evaluation history.
model_evaluation_loss, model_evaluation_accuracy = model_evaluation_history

# Define the date and time format.
date_time_format = '%Y_%m_%d_%H_%M_%S'

# Obtain the current date and time.
current_date_time_dt = dt.datetime.now()

# Convert the date and time to a string with the specified format.
current_date_time_string = dt.datetime.strftime(current_date_time_dt, date_time_format)

# Construct a unique file name based on date, time, loss, and accuracy.
model_file_name = f'LSTM_model_Date_Time_{current_date_time_string}___Loss_{model_evaluation_loss}___Accuracy_{model_evaluation_accuracy}.h5'

# Save the LSTM model with the generated file name.
LSTM_model.save(model_file_name)
```

## Streamlit User-Friendly Version
I have also developed a Streamlit one-click version of the Speech Emotion Recognition model, making it incredibly user-friendly. With this version, users can effortlessly recognize emotions in a speech by simply clicking a single button to upload an audio.
To explore the Streamlit version, click this button below:

[Speech Emotion Recognition](https://speech-emotion-recognition.streamlit.app/)
