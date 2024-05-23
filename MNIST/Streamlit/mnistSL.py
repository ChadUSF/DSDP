import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import transforms
       
app_mode = st.sidebar.selectbox('Select Page',['Home','PyTorch','Tensorflow','Discussion','Numeric Recognition!'])
if app_mode=='Home':
    file_ = open("nums01.gif", "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()
    image_style = """
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 80%;
    """
    st.title('MNIST - Reconizing Handwritten Numerals')
    st.markdown(f'<img src="data:image/gif;base64,{data_url}" alt="cat gif" style="{image_style}">',
             unsafe_allow_html=True,) 
    st.write("""
    ## The Challenge 
    **source:** [Kaggle](https://www.kaggle.com/competitions/digit-recognizer/overview)
             
    MNIST ("Modified National Institute of Standards and Technology") is the de facto “hello world” dataset of computer vision. 
    Since its release in 1999, this classic dataset of handwritten images has served as the basis for benchmarking classification 
    algorithms. As new machine learning techniques emerge, MNIST remains a reliable resource for researchers and learners alike.

    In this competition, your goal is to correctly identify digits from a dataset of tens of thousands of handwritten images. 
    We’ve curated a set of tutorial-style kernels which cover everything from regression to neural networks. 
    We encourage you to experiment with different algorithms to learn first-hand what works well and how techniques compare.
    """)
    st.markdown('## What Does MNIST Look Like?')
    st.image('nums01.png')
   
elif app_mode =='PyTorch':
    file_ = open("nums02.gif", "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()
    image_style = """
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 30%;
    """
    st.title('How a PyTorch CNN Model Performs')
    st.markdown(f'<img src="data:image/gif;base64,{data_url}" alt="cat gif" style="{image_style}">',
             unsafe_allow_html=True,) 
    st.write('''Test accuracy: 0.9908
             This is a great score! We used just over 100 lines of python code to get there.

             The model was created with 20 epochs and batch sizes of 10,000.
            ''')
    st.image('pytorch01.png')
    st.image('pytorch02.png')

    st.write(' ')

    st.write('''Alternatively, running 10 epochs with batches of 1000 was faster but resulted in test accuracy of only 71%.
             ''')
    st.image('pytorch03.png')

elif app_mode =='Tensorflow':
    file_ = open("UC.gif", "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()
    image_style = """
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 80%;
    """
    st.markdown(f'<img src="data:image/gif;base64,{data_url}" alt="cat gif" style="{image_style}">',
             unsafe_allow_html=True,) 
            
elif app_mode =='Discussion':
    file_ = open("UC.gif", "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()
    image_style = """
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 80%;
    """
    st.markdown(f'<img src="data:image/gif;base64,{data_url}" alt="cat gif" style="{image_style}">',
             unsafe_allow_html=True,) 

elif app_mode =='Numeric Recognition!':
    file_ = open("UC.gif", "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()
    image_style = """
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 80%;
    """
    st.markdown(f'<img src="data:image/gif;base64,{data_url}" alt="cat gif" style="{image_style}">',
             unsafe_allow_html=True,) 