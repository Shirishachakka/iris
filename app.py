import streamlit as st
st.title("IRIS-classifier")

sl = st.slider('Sepal Length',4.2,7.9,0.5)
sw = st.slider('Sepal Width',2.2,4.4,0.5)
pl = st.slider('Petal Length',3.5,6.9,0.5)
pw = st.slider('Petal Width',1.5,2.5,0.1)

from sklearn.datasets import load_iris
iris = load_iris()

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(iris.data,iris.target)

op = model.predict([[sl,sw,pl,pw]])
op = iris.target_names[op[0]]
st.title(op)
    
