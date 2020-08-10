#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 23:01:51 2020

@author: mohan
"""



import numpy as np
import pickle
import pandas as pd
#from flasgger import Swagger
import streamlit as st 

from PIL import Image

#app=Flask(__name__)
#Swagger(app)

pickle_in = open("finalized_model.pkl","rb")
classifier=pickle.load(pickle_in)

#@app.route('/')
def welcome():
    return "Welcome All"

#@app.route('/predict',methods=["Get"])
def predict_iris_flower_type(SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm ):
    
    """Let's Authenticate the Banks Note 
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: SepalLengthCm
        in: query
        type: number
        required: true
      - name: SepalWidthCm
        in: query
        type: number
        required: true
      - name: PetalLengthCm
        in: query
        type: number
        required: true
      - name: PetalWidthCm
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values
        
    """
   
    prediction=classifier.predict([[SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm]])
    print(prediction)
    return prediction



def main():
    st.title("PREDICT IRIS FLOWER TYPES")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit predict Iris flower Type ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    SepalLengthCm = st.text_input("SepalLengthCm","Type Here")
    SepalWidthCm = st.text_input("SepalWidthCm","Type Here")
    PetalLengthCm = st.text_input("PetalLengthCm","Type Here")
    PetalWidthCm = st.text_input("PetalWidthCm","Type Here")
    result=""
    if st.button("Predict"):
        result=predict_iris_flower_type(SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm )
    st.success('The output is {}'.format(result))
    if st.button("About"):
        st.text("Lets LEarn")
        st.text("Built with Streamlit")

if __name__=='__main__':
    main()
    