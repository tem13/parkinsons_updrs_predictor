# Test file: pva_0709837_2014-01-21-234832.wav (45M) (35)
# # # # # #
# Imports #
# # # # # #

# Import streamlit
import streamlit as st

# Import file handling
import os
import joblib

# Import dataframe handling
import pandas as pd

# Import feature extraction tools
import parselmouth
from parselmouth.praat import call

# Import audio tools
# import sounddevice as sd
from scipy.io.wavfile import write
from pydub import AudioSegment

# # # # # #
# XGBOOST #
# # # # # #

st.set_page_config(
    page_title='XGBoost',
    page_icon='📈',
)

st.title('XGBoost')

# Loading models
model_male = joblib.load('saved_models/xgboost_male')
model_female = joblib.load('saved_models/xgboost_female')

# Loading scalers
scaler_male = joblib.load('saved_scalers/xgboost_male')
scaler_female = joblib.load('saved_scalers/xgboost_female')

# Model explanation
# st.markdown('This is an explanation of our model.')

# Model metrics
st.header('Metrics')
col1, col2 = st.columns(2)
with col1:
    st.subheader('Male')
    st.markdown('R2_Score: 0.9038128322405257')
    st.markdown('MAE: 2.166250101843663')
    st.markdown('MSE: 8.609510825774194')
    st.markdown('RMSE: 2.9040892804148264')
with col2:
    st.subheader('Female')
    st.markdown('R2_Score: 0.9060637578606281')
    st.markdown('MAE: 2.2711339064800242')
    st.markdown('MSE: 8.319370597285047')
    st.markdown('RMSE: 2.8784916064363575')

# Data processing
train_features = ['age', 'Jitter(%)', 'Jitter(Abs)', 'Jitter:RAP', 'Jitter:PPQ5', 'Jitter:DDP',
                    'Shimmer', 'Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'Shimmer:APQ11',
                    'Shimmer:DDA', 'HNR']

def extract(path, f0min, f0max):
    # Read the sound
    sound = parselmouth.Sound(path)

    # Calculate features
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)
    pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
    localJitter = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    localabsoluteJitter = call(pointProcess, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
    rapJitter = call(pointProcess, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
    ppq5Jitter = call(pointProcess, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
    ddpJitter = call(pointProcess, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)
    localShimmer =  call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    localdbShimmer = call([sound, pointProcess], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq3Shimmer = call([sound, pointProcess], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq5Shimmer = call([sound, pointProcess], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq11Shimmer =  call([sound, pointProcess], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    ddaShimmer = call([sound, pointProcess], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

    # Create dataframe
    data = {'Jitter(%)': localJitter, 'Jitter(Abs)': localabsoluteJitter, 'Jitter:RAP': rapJitter,
            'Jitter:PPQ5': ppq5Jitter, 'Jitter:DDP': ddpJitter, 'Shimmer': localShimmer,
            'Shimmer(dB)': localdbShimmer, 'Shimmer:APQ3': apq3Shimmer, 'Shimmer:APQ5': apq5Shimmer,
            'Shimmer:APQ11': apq11Shimmer, 'Shimmer:DDA': ddaShimmer, 'HNR': hnr}
    df = pd.DataFrame(data, index=[0])
    return df

def normalize_data(data, features, sex):
    if (sex == 'Male'):
        normalized = scaler_male.transform(data[features])
    else:
        normalized = scaler_female.transform(data[features])
    return normalized

# Predicting scores
def predict(sex, data_normalized):
    if (sex == 'Male'):
        model = model_male
    else:
        model = model_female
    prediction = model.predict(data_normalized)
    return prediction

# # # # # #
# Testing #
# # # # # #

# Play audio
def play_audio(path):
    audio_file = open(path, 'rb')
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format='audio/wav')

# Upload and save file
def save_uploadedfile(uploadedfile):
    path = os.path.join(uploadedfile.name)
    with open(path, 'wb') as f:
        f.write(uploadedfile.getbuffer())
        return path

# Slicing audio file
def cut_file(path):
    full = AudioSegment.from_file(path, format='wav')
    segment = full[0:len(full)//2]
    segment.export(path, format='wav')

# Predict sample
def predict_sample(path, age, sex):
    # Preprocess input
    cut_file(path)

    # Extract features
    features_df = extract(path, 75, 500)
    features_df.insert(0, 'age', age)
    features_df.insert(1, 'sex', sex)

    # Delete file
    # os.remove(filename)

    # Normalize data
    data_normalized = normalize_data(features_df, train_features, sex)

    # Predict scores
    prediction = predict(sex, data_normalized)
    st.subheader('Your predicted total_UPDRS score: ' + str(prediction[0,1]))
    st.subheader('Your predicted motor_UPDRS score: ' +  str(prediction[0,0]))

# Forms
def upload_form():
    datafile = st.file_uploader('Upload a voice recording',type=['wav'])
    if datafile is not None:
        filePath = save_uploadedfile(datafile)

        play_audio(filePath)

        with st.form('form'):
            age = st.number_input('Enter your age', step=1)
            sex = st.selectbox('Select your sex', ('Male', 'Female'))
            
            upload_form_submitted = st.form_submit_button('Submit')

            if upload_form_submitted:
                predict_sample(filePath, age, sex)

# def record_form(path):
#     if os.path.exists(path):
#         play_audio(path)

#         with st.form('form'):
#             age = st.number_input('Enter your age', step=1)
#             sex = st.selectbox('Select your sex', ('Male', 'Female'))
            
#             record_form_submitted = st.form_submit_button('Submit')

#             if record_form_submitted:
#                 predict_sample(path, age, sex)

# Try it out!
st.header('Try it out!')
upload_form()

# recording_type = st.selectbox('Choose a method to submit a sample', ('Record now', 'Upload a file'))
# if (recording_type == 'Record now'):
#     filePath = 'recording.wav'
    
#     FS = 44100 # sample rate
#     DURATION = 10

#     if st.button('Record your voice'):
#         try:
#             myrecording = sd.rec(int(DURATION * FS), samplerate=FS, channels=2)
#             with st.spinner('Say "aaah..." at a constant pitch and volume! Recording for 10 seconds...'):
#                 sd.wait()  # Wait until recording is finished
#             st.success('Recording finished!')
#             write(filePath, FS, myrecording)
#             record_form(filePath)
#         except:
#             pass
# else:
#     upload_form()
