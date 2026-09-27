import streamlit as st
import numpy as np
import joblib
model = joblib.load('reg_model_laptop_price')
df = joblib.load('laptop_data')

st.title("Laptop Price Predictor App")
st.caption("Please provide the specification as listed below and click on Predict Price")
st.caption("This ML model is based on 1200 odd laptops only, so might not give the most accurate predictions")

company = st.selectbox("Manufacturer of the laptop",df['Company'].unique(),index=4)
typename = st.radio("Type of the laptop",df['TypeName'].unique(),index=1,horizontal=True)
cpu = st.selectbox("Processor on the laptop",df['Cpu'].unique())
ram = st.radio("RAM on the laptop(in GB)",[4,8,12,16,24,32,64,128],index=1,horizontal=True)
gpu = st.selectbox("Graphics Card",df['Gpu'].unique(),index=1)
os = st.selectbox("Operating System",df['OpSys'].unique(),index=2)
weight = st.slider("Weight of the laptop(in kg)",min_value=0.7,max_value=5.0,step=0.1,value=1.8)
touchscreen = st.radio("Touchscreen",['Yes','No'],index=1)
ips = st.radio("IPS Display",['Yes','No'],index=1,horizontal=True)
cpu_speed = st.slider("CPU Clock Speed(in GHz)",min_value=0.9,max_value=3.9,step=0.1,value=2.2)
hdd = st.radio("Hard Drive(in GB). Select 0 if only SSD is present",[0,250,500,1000,2000],index=0)
ssd = st.radio("SSD Storage(in GB). Select 0 if only HDD is present",[0,128,250,500,1000],index=3)
screensize=st.slider("Screensize(Measured diagonally in inches)",min_value=9.5,max_value=18.5,step=0.1,value=15.6)
screen_resolution = st.selectbox("Screen resolution",[
    "2560x1600","1440x900","1920x1080","2880x1800","1366x768","2304x1440","3200x1800",
    "1920x1200","2256x1504","3840x2160","2160x1440","2560x1440","1600x900","2736x1824",
    "2400x1600"],index=2)

if st.button("Predicate_Price"):
    X_res= int(screen_resolution.split('x')[0])
    Y_res= int(screen_resolution.split('x')[1])
    ppi = round(np.sqrt(X_res**2 + Y_res**2)/screensize)
    if touchscreen == 'YES':
        touchscreen = 1
    else :
        touchscreen = 0
    if ips == 'Yes':
        ips = 1
    else:
        ips = 0
    query= np.array([[company,typename,cpu,ram,gpu,os,weight,touchscreen,ips,cpu_speed,hdd,ssd,ppi]])
    op=model.predict(query)
    st.subheader(f"The estimated price of teh laptof for given specfication ₹{int(round(op[0],-2))}")


