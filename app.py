import numpy as np
import pandas as pd
import tensorflow as tf
import streamlit as st
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder  
from keras.models import load_model
import pickle

#load the trained model
model=load_model("model.h5")

#load the encoders and scalars
with open ("le_encoder_gender.pkl","rb") as file:
    le_encoder_gender=pickle.load(file)

with open ("onehot_encoder_geo.pkl","rb") as file:
    onehot_encoder_geo=pickle.load(file)

with open ("scaler.pkl","rb") as file:
    scaler=pickle.load(file)

#Creating streamlit web app

#1.Title of app
st.title("Churn Modeling Prediction")

# 2.adding user input 
geography=st.selectbox("Geography",onehot_encoder_geo.categories_[0])
gender=st.selectbox("Gender",le_encoder_gender.classes_)
age=st.slider("Age",18,92)
balance=st.number_input("Balance")
credit_score=st.number_input("Credit Score")
estimated_salary=st.number_input("Estimated Salary")
tenure=st.slider("Tenure",0,10)
num_Of_products=st.slider("NumOfProducts",1,4)
has_cr_card	=st.selectbox("HasCrCard",[0,1])
is_active_member=st.selectbox("IsActiveMember",[0,1])

#Preparing the input data
input_data=pd.DataFrame({
"Geography": [geography],
"Credit Score":[credit_score],
"Gender":[le_encoder_gender.transform([gender])[0]],
"Age" : [age],
"Tenure":[tenure],
"Balance":[balance],
"NumOfProducts":[num_Of_products],
"HasCrCard":[has_cr_card],
"IsActiveMember":[is_active_member],
"EstimatedSalary":[estimated_salary],
})

#onehot encoded geography
geo_encoder=OneHotEncoder()
geo_encoded=geo_encoder.fit_transform(input_data[["Geography"]])
# geo_columns=geo_encoder.get_feature_names_out("Geography")
geo_columns = geo_encoder.get_feature_names_out("Geography")
geo_encoded_df=pd.DataFrame(geo_encoded.toarray(),columns=geo_columns)

#combine one hot encoded columns with input data
input_data=pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)

#Scale the input data
input_data_scaled=scaler.transform(input_data)

#prediction churn
prediction=model.predict(input_data_scaled)
prediction_proba=prediction [0][0]

if prediction_proba>0.5:
    st.write("The customer is likely to churn")
else:
    st.write("The customer is not likely to churn")