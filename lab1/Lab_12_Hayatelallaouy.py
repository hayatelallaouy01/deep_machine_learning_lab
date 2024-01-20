# Lab_12 : Classification des fleurs iris
# Réalisé par : Hayat el allaouy /Emsi 2023-2024
# Library import
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
import pandas as pd


# Step 1: DataSet
iris = datasets.load_iris()
print(iris.data)
print(iris.feature_names) # Colonnes
print(iris.target)
print(iris.target_names)
print(iris.data.shape)

# Step 2: Model
selected_model = 'RandomForest'
if selected_model == 'RandomForest' :
    model = RandomForestClassifier()



# Step 3: Train
model.fit(iris.data, iris.target)

# Step 4: Test
prediction = model.predict([[5.0, 3.9, 0.1, 0.1]])
print(prediction)
print(iris.target_names[prediction])

# Model Deployment with streamlit: streamlit run Lab_12_AghzerHousna.py
st.header('iris classification model')
st.image('images/iris_categories.png')
# st.write(iris.data)
# st.write(iris.feature_names)
def user_input():
    sepal_length = st.sidebar.slider('sepal length', 4.3, 7.9, 6.0)
    sepal_width = st.sidebar.slider('sepal width', 2.0, 4.4, 3.0)
    petal_length = st.sidebar.slider('petal length', 1.0, 9.2, 2.0)
    petal_width = st.sidebar.slider('petal width', 0.1, 2.5, 1.0)
    data = {
        'sepal_length':sepal_length,
        'sepal_width':sepal_width,
        'petal_length':petal_length,
        'petal_width':petal_width
    }
    flower_features = pd.DataFrame(data,index=[0])
    return flower_features



st.sidebar.header('Iris features')
df = user_input()
st.write(df)
st.subheader('Prediction')
prediction = model.predict(df)
st.write(iris.target_names[prediction])
st.image("images/"+iris.target_names[prediction][0]+".png")
selected_model = st.sidebar.selectbox('Select a learning model', ['RandomForest', 'Decisiontree', 'KNN', 'SVM'])
st.write('Selected model is : ', selected_model)



