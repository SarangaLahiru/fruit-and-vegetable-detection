import streamlit as st
import tensorflow as tf
import numpy as np

def model_prediction(test_image):
    model= tf.keras.models.load_model("trained_model.h5")
    image=tf.keras.preprocessing.image.load_img(test_image,target_size=(64,64))
    input_arr=tf.keras.preprocessing.image.img_to_array(image)
    input_arr=np.array([input_arr])
    predictions=model.predict(input_arr)
    return np.argmax(predictions)

st.sidebar.title("Dashboard")

app_mode=st.sidebar.selectbox("select page",["Home","About project","Prediction"])


if(app_mode=="Home"):
    st.header("Fruit and vegitable Recongnize System")
    

elif(app_mode=="About project"):
    st.header("About Project")

elif(app_mode=="Prediction"):
    st.header("Model Prediction")
    test_image=st.file_uploader("Choose as Image")
    
    if(st.button("Show Image")):
        st.image(test_image)

    if(st.button("Predict")):
        st.write("Prediction")
        result_index=model_prediction(test_image)

        with open("labels.txt") as f:
            content=f.readlines()
        label=[]
        
        for i in content:
            label.append(i[:-1])

        st.success("It is "+format(label[result_index]))
        

    
