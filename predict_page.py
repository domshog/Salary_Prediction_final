import streamlit as st
import pickle
import numpy as np

def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data= load_model ()

regressor =data["model"]
le_country =data["le_country"]
le_education =data["le_education"]

def show_predict_page():
    st.title("Software Develpoer Salary Prediction")

    st.write("""### We need some info to predict the salary""")

    countries=(

        "United States of America",
        "Germany",
        "United Kingdom of Great Britain and Northern Ireland",
        "Canada",
        "Norway",
        "Denmark",
        "Switzerland",
        "Poland",
        "Italy",
        "Sweden",
        "Spain",
        "Brazil",
        "Australia",
        "Netherlands",
        "France",
        "India",



    )

    Education=(

        "Bachelor’s degree",
        "Master’s degree",
        "Post grad",
        "Less than a Bachelors"

    )

    country=st.selectbox("Country", countries)
    Education_Level =st.selectbox("Eduction Level", Education)
    experience= st.slider("Years of Experience", 0, 50, 3)
    ok=st.button("Calculate Salary")
    if ok:
        X= np.array([[country, Education_Level, experience ]])
        X[:, 0] = le_country.transform(X[:,0])
        X[:, 1] = le_education.transform(X[:,1])
        X = X.astype(float)

        salary =regressor.predict(X)
        st.subheader(f"The estimated salary is ${salary[0]:.2f}")
