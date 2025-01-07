import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import os

# Sidebar navigation
r = st.sidebar.radio('Main Menu', ['Home', 'Depression Prediction'])

# Home Page
if r == 'Home':
    st.title('DEPRESSION PREDICTION')
    st.subheader("It’s okay to not be okay. Let’s take this journey together.")
    st.markdown("*You can predict your depression using ANN Deep Learning on the next page* 😎")
    st.image("depSigns.PNG",'rb')

# Depression Prediction Page
elif r == 'Depression Prediction':
    left_column, right_column = st.columns(2)

    # Collect user inputs
    gender = left_column.selectbox("Please select your gender", ('Male', 'Female'))
    p1 = 1 if gender == "Male" else 0

    p2 = right_column.slider("What is your age?", 0, 100)

    marital_status = left_column.selectbox(
        "Select your marital status", 
        ('Married', 'Never Married', 'Widowed', 'Divorced', 'Separated', 'Partner')
    )
    marital_map = {'Married': 1, 'Never Married': 2, 'Widowed': 5, 'Divorced': 0, 'Separated': 4, 'Partner': 3}
    p3 = marital_map[marital_status]

    pregnant = right_column.selectbox("Are you pregnant?", ('Yes', 'No'))
    p4 = 1 if pregnant == "Yes" else 0

    p5 = left_column.slider("What is your BMI?", 0, 50)

    trouble_sleeping = right_column.selectbox("Do you have trouble sleeping?", ('Yes', 'No'))
    p6 = 1 if trouble_sleeping == "Yes" else 0

    p7 = left_column.slider("How many hours do you sleep?", 0, 24)

    vigorous_recreation = right_column.selectbox(
        "Do you do vigorous recreation like running, cycling, swimming etc?", ('Yes', 'No')
    )
    p8 = 1 if vigorous_recreation == "Yes" else 0

    moderate_recreation = left_column.selectbox(
        "Do you do moderate recreation like walking, yoga, gardening etc?", ('Yes', 'No')
    )
    p9 = 1 if moderate_recreation == "Yes" else 0

    vigorous_work = right_column.selectbox(
        "Does your work come under vigorous work like construction work, farming?", ('Yes', 'No')
    )
    p10 = 1 if vigorous_work == "Yes" else 0

    moderate_work = left_column.selectbox(
        "Does your work come under moderate work like desk job?", ('Yes', 'No')
    )
    p11 = 1 if moderate_work == "Yes" else 0

    alcohol = right_column.selectbox("Do you consume alcohol?", ('Yes', 'No'))
    p12 = 1 if alcohol == "Yes" else 0

    memory_problems = left_column.selectbox("Do you have memory problems?", ('Yes', 'No'))
    p13 = 1 if memory_problems == "Yes" else 0

    smoking = right_column.selectbox("Do you smoke?", ('Yes', 'No'))
    p14 = 1 if smoking == "Yes" else 0

    health_issues = left_column.multiselect(
        "Do you have any health problems?",
        [
            'Other Impairment', 'Bone or Joint', 'Weight', 'Back or Neck', 'Arthritis',
            'Cancer', 'Other Injury', 'Breathing', 'Stroke', 'Blood Pressure',
            'Mental Retardation', 'Hearing', 'Heart', 'Vision', 'Diabetes',
            'Birth Defect', 'Senility', 'Other Developmental'
        ]
    )

    # Map health issues to binary features
    health_conditions = [
        'Other Impairment', 'Bone or Joint', 'Weight', 'Back or Neck', 'Arthritis',
        'Cancer', 'Other Injury', 'Breathing', 'Stroke', 'Blood Pressure',
        'Mental Retardation', 'Hearing', 'Heart', 'Vision', 'Diabetes',
        'Birth Defect', 'Senility', 'Other Developmental'
    ]
    health_data = [1 if condition in health_issues else 0 for condition in health_conditions]

    # Load the model
    model_path = 'model.h5'
    if not os.path.exists(model_path):
        st.error("Model file not found. Please check the path.")
    else:
        model = load_model(model_path)

        # Prepare input data
        input_data = np.array([[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14] + health_data])

        # Ensure input matches the model's expected shape
        expected_features = model.input_shape[-1]
        if input_data.shape[1] < expected_features:
            # Pad with zeros
            padded_input = np.zeros((1, expected_features))
            padded_input[0, :input_data.shape[1]] = input_data
            input_data = padded_input
        elif input_data.shape[1] > expected_features:
            # Trim to required size
            input_data = input_data[:, :expected_features]

        # Predict and display the result
        if st.button('Predict'):
            pred = model.predict(input_data)
            predicted_class = np.argmax(pred, axis=1)[0]
            if predicted_class == 0:
                st.success("Congratulations! You do not have depression.")
            else:
                st.warning("Your chances of having depression are high! Please consult a doctor.")
