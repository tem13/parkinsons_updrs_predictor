
# # # # # #
# Imports #
# # # # # #

# Import streamlit
import streamlit as st

# Import dataframe handling
import pandas as pd

# # # #
# APP #
# # # #

st.set_page_config(
    page_title='UPDRS Score Predictor',
)

st.title("Parkinson's UPDRS Score Predictor")
st.caption('2022 Medlytics; Christine Tian, Daniel Kim, Harshini Magesh, Shreya Singh, Sreeja Challa, Tem Taepaisitphongse')

# Presentation
# st.header('Presentation')
# st.video('video url')

# Loading data
DATA_URL = '../data/parkinsons_updrs.data'

@st.cache
def load_data():
    data = pd.read_csv(DATA_URL)
    return data

data = load_data()

# Display data
st.header('Data')
st.markdown('Taken from the UCI Machine Learning Repository\nhttps://archive.ics.uci.edu/ml/datasets/Parkinsons+Telemonitoring')
if st.checkbox('Show raw data', value=False):
    sex = st.radio('Sex', ('All', 'Male', 'Female'))
    if sex == 'Male':
        st.subheader('Raw data: male')
        st.write(data[data['sex'] == 0])
    elif sex == 'Female':
        st.subheader('Raw data: female')
        st.write(data[data['sex'] == 1])
    else:
        st.subheader('Raw data - male & female')
        st.write(data)
