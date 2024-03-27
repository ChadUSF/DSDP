import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
#import pickle  # to load a saved model
#import base64  # to encode .gif files for HTML embedding

# Define helper functions
def get_fvalue(val, feature_dict={"No": 0, "Yes": 1}):
    return feature_dict.get(val, None)

def get_dependents_value(Dependents):
    dependents_dict = {'0': 0, '1': 1, '2': 2, '3+': 3}
    return dependents_dict.get(Dependents, 0)

#def load_model(model_path='Random_Forest.sav'):
    #with open(model_path, 'rb') as file:
        #loaded_model = pickle.load(file)
    #return loaded_model

#def encode_gif(gif_path):
    #with open(gif_path, "rb") as file:
        #contents = file.read()
        #data_url = base64.b64encode(contents).decode("utf-8")
    #return data_url

# App starts here
app_mode = st.sidebar.selectbox('Select Page', ['Home', 'Prediction'])  # two pages

if app_mode == 'Home':
    st.title('LOAN PREDICTION APP')      
    st.image('loan_image.jpg')    
    st.markdown('## Dataset Overview')    
    data = pd.read_csv('loan_dataset.csv')    
    st.dataframe(data.head())    
    st.markdown('## Applicant Income VS Loan Amount')    
    st.bar_chart(data[['ApplicantIncome', 'LoanAmount']].head(20))

elif app_mode == 'Prediction':
    st.image('slider-short-3.jpg')    
    st.subheader('Please fill all necessary information to get a reply to your loan request!')    
    
    # Sidebar form for input features
    with st.sidebar.form(key='loan_form'):
        st.header("Client Information:")    
        gender_dict = {"Male": 1, "Female": 2}    
        edu = {'Graduate': 1, 'Not Graduate': 2}    
        prop = {'Rural': 1, 'Urban': 2, 'Semiurban': 3}
        feature_dict = {"No": 0, "Yes": 1}  # Adjusted the mapping to be consistent

        # Form fields
        ApplicantIncome = st.slider('ApplicantIncome', 0, 10000, 5000)    
        CoapplicantIncome = st.slider('CoapplicantIncome', 0, 10000, 2500)    
        LoanAmount = st.slider('LoanAmount (in $K)', 9.0, 700.0, 200.0)    
        Loan_Amount_Term = st.selectbox('Loan_Amount_Term (in months)', (12, 36, 60, 84, 120, 180, 240, 300, 360))    
        Credit_History = st.radio('Credit_History', (0.0, 1.0))    
        Gender = st.radio('Gender', tuple(gender_dict.keys()))    
        Married = st.radio('Married', tuple(feature_dict.keys()))    
        Self_Employed = st.radio('Self Employed', tuple(feature_dict.keys()))    
        Dependents = st.radio('Dependents', ('0', '1', '2', '3+'))    
        Education = st.radio('Education', tuple(edu.keys()))    
        Property_Area = st.radio('Property_Area', tuple(prop.keys()))    
        submit_button = st.form_submit_button(label='Predict')

    # Prediction logic
    if submit_button:
        # Encode input features
        feature_vector = np.array([
            ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History,
            get_fvalue(Gender, gender_dict), get_fvalue(Married), get_fvalue(Self_Employed),
            edu[Education], prop[Property_Area],  # Using direct mapping for simplicity
            get_dependents_value(Dependents)
        ]).reshape(1, -1)
        
    # Build model and make prediction
        data = pd.read_csv('loan_dataset.csv')
        label_encoder = LabelEncoder()
        categorical_columns = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Dependents']
        for col in categorical_columns:
            data[col] = label_encoder.fit_transform(data[col].astype(str))

        X = data.drop(['Loan_Status','Loan_ID'], axis=1)
        y = data['Loan_Status']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)
        model = RandomForestClassifier(n_estimators=10, max_depth=4)
        model.fit(X_train, y_train)
        prediction = model.predict(feature_vector)
        
        # Display results
        if prediction[0] == 0:
            st.error('According to our Calculations, you will not get the loan from Bank')            
            st.image('slider-short-3.jpg')    
        elif prediction[0] == 1:            
            st.success('Congratulations!! you will get the loan from Bank')            
            st.image('rain.jpg')