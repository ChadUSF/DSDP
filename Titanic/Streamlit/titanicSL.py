import streamlit as st
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import pickle
       
app_mode = st.sidebar.selectbox('Select Page',['Home','Visuals','Prediction'])
if app_mode=='Home':
    file_ = open("titanic.gif", "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()
    image_style = """
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 80%;
    """
    st.title('Titanic - Machine Learning from Disaster')
    st.markdown(f'<img src="data:image/gif;base64,{data_url}" alt="cat gif" style="{image_style}">',
             unsafe_allow_html=True,) 
    st.write("""
    ## The Challenge 
    **source:** [Kaggle](https://www.kaggle.com/c/titanic)

    The sinking of the Titanic is one of the most infamous shipwrecks in history.

    On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding with an iceberg. Unfortunately, there weren’t enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew.

    While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others.

    In this challenge, we ask you to build a predictive model that answers the question: “what sorts of people were more likely to survive?” using passenger data (ie name, age, gender, socio-economic class, etc).

    ### What Data Will I Use in This Competition?
    In this competition, you’ll gain access to two similar datasets that include passenger information like name, age, gender, socio-economic class, etc. One dataset is titled train.csv and the other is titled test.csv.

    Train.csv will contain the details of a subset of the passengers on board (891 to be exact) and importantly, will reveal whether they survived or not, also known as the “ground truth”.

    The test.csv dataset contains similar information but does not disclose the “ground truth” for each passenger. It’s your job to predict these outcomes.

    Using the patterns you find in the train.csv data, predict whether the other 418 passengers on board (found in test.csv) survived.

    """)
    
   
elif app_mode =='Visuals':
    file_ = open("titanic2.gif", "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()
    image_style = """
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 80%;
    """
    data = pd.read_csv('titanic.csv') 
    sel_cols = ['Pclass', 'Age', 'Fare','SibSp','Parch','Survived']
    sum_data = data[sel_cols].describe()
    st.markdown(
        f'<img src="data:image/gif;base64,{data_url}" alt="cat gif" style="{image_style}">',
        unsafe_allow_html=True,
    )
    st.markdown('### Titanic Dataset Overview')    
    st.dataframe(data.head())

    st.markdown('### Summary Statistics')
    st.write(sum_data)

    st.markdown('## Gender and Survival Rate')
    survival_by_sex = data.groupby('Sex')['Survived'].mean()
    st.bar_chart(survival_by_sex)

    st.markdown('## Gender and Age with Survival')
    survived = 'survived'
    not_survived = 'not survived'
    women = data[data['Sex'] == 'female']
    men = data[data['Sex'] == 'male']
    fig2, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    ax = sns.histplot(women[women['Survived'] == 1].Age.dropna(), bins=18, label=survived, ax=axes[0], kde=False)
    ax = sns.histplot(women[women['Survived'] == 0].Age.dropna(), bins=40, label=not_survived, ax=axes[0], kde=False)
    ax.legend()
    ax.set_title('Female')
    ax = sns.histplot(men[men['Survived'] == 1].Age.dropna(), bins=18, label=survived, ax=axes[1], kde=False)
    ax = sns.histplot(men[men['Survived'] == 0].Age.dropna(), bins=40, label=not_survived, ax=axes[1], kde=False)
    ax.legend()
    _ = ax.set_title('Male')
    st.pyplot(fig2)

    st.markdown('## Passenger Class and Survival Rate')
    survival_by_Pc = data.groupby('Pclass')['Survived'].mean()
    fig, ax = plt.subplots()
    survival_by_Pc.plot(kind='bar', ax=ax)
    plt.xlabel('Passenger Class')
    plt.ylabel('Survival Rate')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    st.pyplot(fig)
    
    st.markdown('## Count of Passenger Class')
    st.bar_chart(data['Pclass'].value_counts())

elif app_mode =='Prediction':
    st.image('survive.jpg',width=500)
    st.subheader('Complete criteria to determine if you would perish like Jack or survive like Rose!')
    st.sidebar.header("Passengar Information:")
    gender_dict = {"Male":'male',"Female":'female'}
    embarked_dict = {'Cherbourg': 'C', 'Queenstown': 'Q', 'Southampton': 'S'}
    Pclass = st.sidebar.radio('Passenger Class:', options=['1', '2', '3'])
    Sex = st.sidebar.radio('Gender:', tuple(gender_dict.keys()))
    Age = st.sidebar.slider('Age:', 0, 100, 30)
    Fare = st.sidebar.slider('Fare:', 0, 150, 33)
    SibSp = st.sidebar.slider('\# of siblings / spouses aboard the Titanic', 0, 10, 1)
    Parch = st.sidebar.slider('\# of parents / children aboard the Titanic', 0, 10, 1)
    Embarked = st.sidebar.radio('Embarked From:', tuple(embarked_dict.keys()))

    model = 'SVC.sav'
    loaded_model = pickle.load(open(model, 'rb'))

    train=pd.read_csv('titanic.csv')
    train_inputs=train.drop(['PassengerId','Name', 'Ticket', 'Cabin','Survived'], axis=1)

    numeric_columns = ['Age', 'SibSp', 'Parch', 'Fare']
    categorical_columns = ['Pclass', 'Sex', 'Embarked']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_columns),
        ('cat', categorical_transformer, categorical_columns)
    ])

    train_x = preprocessor.fit_transform(train_inputs)

    def predict_survival(features):
        processed_features = preprocessor.transform(features)
        prediction = loaded_model.predict(processed_features)
        return prediction

    input_features=pd.DataFrame({
        'Pclass':[Pclass],
        'Sex':[Sex],
        'Age':[Age],
        'Fare':[Fare],
        'SibSp':[SibSp],
        'Parch':[Parch],
        'Embarked':[Embarked]
    })

    if st.sidebar.button("Predict"):
        image_style = """
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 80%;
        """
        file1 = open("alive.gif", "rb")
        contents = file1.read()
        data_url = base64.b64encode(contents).decode("utf-8")
        file1.close()
   
        file2 = open("sorry.gif", "rb")
        contents = file2.read()
        data_url2 = base64.b64encode(contents).decode("utf-8")
        file2.close()
        
        file3 = open("sorry2.gif", "rb")
        contents = file3.read()
        data_url3 = base64.b64encode(contents).decode("utf-8")
        file3.close()
   
        prediction = predict_survival(input_features)
        if prediction[0] == 0 :
            st.error(
    'We are sorry to tell you, but you have joined Jack at the bottom of the sea.'
    )
            st.markdown(
    f'<img src="data:image/gif;base64,{data_url2}" alt="cat gif" style="{image_style}">',
    unsafe_allow_html=True,)
            st.markdown(
    f'<img src="data:image/gif;base64,{data_url3}" alt="cat gif" style="{image_style}">',
    unsafe_allow_html=True,)            

        elif prediction[0] == 1 :
            st.success(
    'Congratulations!! You get to live with survivors guilt!'
    )
            st.markdown(
    f'<img src="data:image/gif;base64,{data_url}" alt="cat gif" style="{image_style}">',
    unsafe_allow_html=True,)