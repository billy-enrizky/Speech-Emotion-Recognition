import streamlit as st  # Importing the Streamlit library
import numpy as np  # Importing NumPy for numerical operations
from tensorflow.keras.models import load_model  # Importing Keras for loading the trained model
import os  # Importing the os module for interacting with the operating system
import librosa  # Importing the librosa library for audio processing

@st.cache_resource
def load_LSTM_model():
    '''Load the pre-trained LSTM model from a specific file.'''
    return load_model("LSTM_model_Date_Time_2024_01_03_20_39_00___Loss_0.043416813015937805___Accuracy_0.9861111044883728.h5")

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

def predict(model, wav_filepath):
    '''Predicting the emotion from the given audio file using the loaded model.'''
    emotions = {1 : 'neutral', 2 : 'calm', 3 : 'happy', 4 : 'sad', 
                5 : 'angry', 6 : 'fearful', 7 : 'disgust', 8 : 'surprised'}
    
    # Extracting MFCC features from the WAV file
    test_point = extract_mfcc(wav_filepath)
    
    # Reshaping the input to match the model's expected input shape
    test_point = np.reshape(test_point, newshape=(1, 40, 1))
    
    # Making predictions using the model
    predictions = model.predict(test_point)
    
    # Printing and returning the predicted emotion
    print(emotions[np.argmax(predictions[0]) + 1])
    return emotions[np.argmax(predictions[0]) + 1]

def upload_audio():
    '''Function to handle audio file uploading and emotion prediction.'''
    models_load_state = st.text('\n Loading models...')
    
    # Loading the LSTM model
    model = load_LSTM_model()
    
    # Updating the loading state
    models_load_state.text('\n Models Loading Complete!')
    
    # Displaying a sidebar message for uploading an audio file
    st.sidebar.success('Try Uploading an audio file.')
    
    # Uploading an audio file
    file_uploaded = st.file_uploader("Choose an audio...", type='wav')
    
    if file_uploaded:
        # Displaying the uploaded audio and predicting the emotion
        st.audio(file_uploaded, format='audio/wav')
        st.success('Emotion of the audio is  ' + predict(model, file_uploaded))

def main():
    '''Main function to create the Streamlit app.'''
    st.image('speechrecog.jpeg')  # Displaying an image
    st.title('Speech Emotion Recognition')  # Displaying the title
    upload_audio()  # Calling the function for audio uploading and prediction
    
if __name__ == "__main__":
    main()  # Running the main function when the script is executed
