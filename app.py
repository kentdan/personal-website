import streamlit as st
import pandas as pd
import os

import pandas_profiling 
from streamlit_pandas_profiling import st_profile_report

from pycaret.regression import setup, compare_models, pull, save_model
with st.sidebar:
    #st.image('')
    st.title('Eddy Covariance')
    choice = st.radio('navigation',['Upload','Profiling','ML','Download'])
    st.info('this app allow you to build automated ML pipeline using pandas profiling pycaret')

if os.path.exists('./downloaded/data.csv'):
    df = pd.read_csv('./downloaded/data.csv',index_col=None)

if choice == "Upload":
    st.title('Upload data for modelling')
    file = st.file_uploader('Upload your CSV here')
    if file:
        df = pd.read_csv(file,header=[1],index_col=None) #ignore header 0 header=[1]
        df = df.iloc[2: , :]
        df.to_csv('./downloaded/data.csv',index=None)
        st.dataframe(df)
        

if choice == "Profiling":
    st.title('Automated EDA')
    if st.button('Automated EDA'):
        profile_report = df.profile_report()
        st_profile_report(profile_report)
    

if choice == "ML":
    st.title('Machine Learning Comparison')
    target = st.selectbox('Select Your Target',df.columns)
    if st.button('Train Model'):
        setup(df,target=target,silent=True)
        setup_df = pull()
        st.info(" ML experiment setting")
        best_model = compare_models()
        compare_df = pull()
        st.info(" ML Model")
        st.dataframe(compare_df)
        best_model
        save_model(best_model,'best_model')

if choice == "Download":
    with open('best_model.pkl','rb') as f: 
        st.download_button("Download the model",f,'trained_model.pkl')