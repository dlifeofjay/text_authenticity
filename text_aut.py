import streamlit as st
import joblib
import pandas as pd
import numpy as np
from datetime import datetime


model = joblib.load('text_aut.joblib')
oe = joblib.load('text_OrdEnc.joblib')
cv = joblib.load('text_cv.joblib')


DataFile = 'text_aut.csv'


try:
    df = pd.read_csv(DataFile)
except FileNotFoundError:
    df = pd.DataFrame(columns=['Text', 'Prediction', 'Timestamp'])

st.title('Fake News Detector')
st.write('Check if a news is real or fake')

with st.form('Open News'):
    Message = st.text_input('Enter Your text:')

    submitted = st.form_submit_button('Predict')

    if submitted:
        input_data = pd.DataFrame({'Message': [Message]})  # Ensure structured input
        input_df = cv.transform(input_data['Message'])  # Transform the input message

        prediction = model.predict(input_df)[0]
        label = oe.inverse_transform([[prediction]])[0][0]

        if prediction == 0:
            st.success('This news seems to be true')
        else:
            st.error('This news seems to be fake')

        # Append new data entry
        new_row = pd.DataFrame({'Message': [Message], 'Prediction': [label], 'Timestamp': [datetime.now()]})
        df = pd.concat([df, new_row], ignore_index=True)

        # Save updated data to CSV
        df.to_csv(DataFile, index=False)
        st.info('Prediction Saved to DataFrame')

if st.checkbox('Show Collected Data'):
    st.dataframe(df)

if st.button('Download Collected Data as CSV'):
    st.download_button(
        label='Download CSV',
        data=df.to_csv(index=False).encode('utf-8'),
        file_name='SVM_Email.csv',
        mime='text/csv'
    )
