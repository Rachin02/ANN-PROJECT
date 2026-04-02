import tensorflow
import pickle
import pandas as pd
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model

# source /Users/rachin/Desktop/rachin/ai/bin/activate

# load model
model = load_model('model.h5')

#load encoder and scaler
with open('lr_gender.pkl','rb') as file:
    label_gender = pickle.load(file)

with open('ohe_geo.pkl','rb') as file:
    label_geo = pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)


## streamlit app
st.title('Customer churn prediction')


#user input
geography = st.selectbox('Geography', label_geo.categories_[0])
gender = st.selectbox('Gender',label_gender.classes_)
age = st.slider('age',18,92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit score')
estimated_salary = st.number_input("Estimated salary")
tenure = st.slider('Tenure',0,10)
num_of_products = st.slider('Number of products',0,8)
has_credit = st.selectbox('Has credit card',[0,1])
is_active_member = st.selectbox("Is active member",[0,1])


#prepare input data

input_data = {
    'CreditScore' : [credit_score],
    'Gender' : [gender],
    'Age' : [age],
    'Tenure' : [tenure],
    'Balance' : [balance],
    'NumOfProducts' : [num_of_products],
    'hasCrCard' : [has_credit],
    'isActiveMember' : [is_active_member],
    'Estimated_salary' : [estimated_salary]
}


# One-hot encode 'Geography'
geo_encoded = label_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=label_geo.get_feature_names_out(['Geography']))

# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)


# Predict churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.write(f'Churn Probability: {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')
