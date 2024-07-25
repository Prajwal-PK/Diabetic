import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE

url="https://raw.githubusercontent.com/Prajwal-PK/Diabetic/main/Datasets/Diabeties.csv"
df=pd.read_csv(url)

x=df.iloc[:,:8]
y=df.iloc[:,-1]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2,random_state=20,stratify=y)


logreg=LogisticRegression(max_iter=1000)
logreg.fit(x,y)

st.markdown("""
<div style="text-align: center;">
    <h1>Diabetes Prediction</h1>
    <p>This app predicts whether a person is diabetic or not based on input parameters.</p>
</div>
""", unsafe_allow_html=True)


st.markdown("""
<div style="text-align: center;">
    <h2>*-* Enter following details *-*</h2>
</div>
""", unsafe_allow_html=True)

#Streamlit
st.subheader("No of pregnancies")
preg = st.number_input("", min_value = 0, max_value = 20)
st.subheader("Plasma Glucose Level")
glu = st.number_input("", min_value = 50, max_value = 500)
st.subheader("Diastolic Blood Pressure (mm Hg)")
bp = st.number_input("", min_value = 60, max_value = 300)
st.subheader("Skin Thickness (mm)")
skin = st.number_input("", min_value = 10, max_value = 60)
st.subheader("Insulin (U/ml)")
Insulin = st.number_input("", min_value = 40, max_value = 500)
st.markdown("""<h4>Calculate your BMI</h4>""", unsafe_allow_html= True)
c1, c2 = st.columns(2)
with c1:
    wt = st.number_input("Weight", min_value=30, max_value=150)
with c2:
    ht = st.number_input("Height (cm)", min_value=145, max_value=240)
bmi = wt / ((ht/100)**2)
st.subheader(bmi)
st.write("")
st.subheader("Diabetes Pedigree Function")
dpg = st.number_input("", min_value = 0.01, max_value = 2.0)
st.subheader("Age")
age = st.number_input("", min_value = 16, max_value = 120)

temp=[preg,glu,bp,skin,Insulin,bmi,dpg,age]
df2=pd.DataFrame(columns=x.columns)
df2.loc[0]=temp

pr=logreg.predict(df2)

rs = st.button("Result")
if rs == 1 and pr == 1:
    st.header("Chances of being Diabetic")
elif rs == 1 and pr == 0:
    st.header("Don't worry, NO signs of being Diabetic")
else:
    st.header("")
st.write('---')