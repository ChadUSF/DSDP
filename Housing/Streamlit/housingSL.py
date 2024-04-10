import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import scatterplotmatrix
from mlxtend.plotting import heatmap
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import base64
import pickle
       
app_mode = st.sidebar.selectbox('Select Page',['Home','Visuals','Prediction'])
columns = ['Overall Qual', 'Overall Cond', 'Gr Liv Area',
        'Central Air', 'Total Bsmt SF', 'SalePrice']
df = pd.read_csv('AmesHousing.txt', sep='\t', usecols=columns)
df = df.dropna(axis=0)
df['Central Air'] = df['Central Air'].map({'N': 0, 'Y': 1})
sum_data = df.describe()

target = 'SalePrice'
features = df.columns[df.columns != target]
X = df[features].values
y = df[target].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=123)

forest = RandomForestRegressor(n_estimators=1000, 
                            criterion='squared_error', 
                            random_state=1, 
                            n_jobs=-1)
forest.fit(X_train, y_train)
y_train_pred = forest.predict(X_train)
y_test_pred = forest.predict(X_test)

mae_train = mean_absolute_error(y_train, y_train_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)
r2_train = r2_score(y_train, y_train_pred)
r2_test =r2_score(y_test, y_test_pred)

if app_mode=='Home':
    st.image('hv1.jpg',width=500)
    st.title('Using Regression to Predict Home Value')
    st.write("""
    # Housing data Regression
    This is a streamlit implementation using a sample housing dataset, working through example regression exercises from "Machine Learning with PyTorch and Scikit-Learn" text.
    
    """)
   
elif app_mode =='Visuals': 
    st.markdown('### Random Forest Regression Results') 
    st.write(f'MAE train: {mae_train:.2f}')
    st.write(f'MAE test: {mae_test:.2f}')
    st.write(f'R^2 train: {r2_train:.2f}')
    st.write(f'R^2 test: {r2_test:.2f}')

    st.markdown('### Random Forest Regression Feature Importance')
    feature_imp = pd.DataFrame(
    {'importance':forest.feature_importances_},
    index=features)
    feature_imp.sort_values(by='importance', ascending=False)
    st.write(feature_imp)

    st.markdown('### Housing Dataset Overview')
    st.dataframe(df.head())

    st.markdown('### Summary Statistics')
    st.write(sum_data)

    st.markdown('## Correlation Heatmap')
    cm = np.corrcoef(df.values.T)
    plt.figure(figsize=(10, 8))
    hm = heatmap(cm, row_names=df.columns, column_names=df.columns, cmap='coolwarm')
    st.pyplot(plt.gcf())

    st.markdown('## Sale Price Distribution')
    plt.figure(figsize=(8, 6))
    df['SalePrice'].plot.hist(bins=20, edgecolor="black")
    plt.xlabel("Sale Price in $")
    st.pyplot(plt.gcf())


elif app_mode =='Prediction':
    st.image('hv2.png',width=700)
    st.subheader('Complete criteria to determine home value.')
    st.sidebar.header("Home Information:")
    ac_dict = {"Yes":1,"No":0}
    OvrQual = st.sidebar.slider('Overall Quality:', 0, 10, 5)
    OvrCond = st.sidebar.slider('Overall Condition:', 0, 10, 5)
    GrLivArea = st.sidebar.slider('Above Grade Living Area:', 500, 6000, 1500)
    BsmtSF = st.sidebar.slider('Total SqFt of Basement Area:', 0, 6000, 1000)
    ac = st.sidebar.radio('Central Air:', tuple(ac_dict.keys()))

    input_data=pd.DataFrame({
        'Overall Qual':[OvrQual],
        'Overall Cond':[OvrCond],
        'Gr Liv Area':[GrLivArea],
        'Central Air':[ac_dict[ac]],
        'Total Bsmt SF':[BsmtSF]
    })

    if st.sidebar.button("Predict"):
        image_style = """
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 80%;
        """
        file = open("money.gif", "rb")
        contents = file.read()
        data_url = base64.b64encode(contents).decode("utf-8")
        file.close()

        pred = forest.predict(input_data)

        st.subheader(f"Predicted Sale Price: ${pred[0]:,.2f}")
