import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import streamlit as st

# Load the pre-trained model
model_filename1 = 'models/random_forest_model.pkl'
with open(model_filename1, 'rb') as file:
    model_random = pickle.load(file)

# Load the label encoders
model_filename2 = 'models/label_encoder.pkl'
with open(model_filename2, 'rb') as f:
    label_encoders = pickle.load(f)

# Load the standard scaler
model_filename3 = 'models/standard_scaler.pkl'
with open(model_filename3, 'rb') as f:
    scaler = pickle.load(f)

columns = ['International plan', 'Voice mail plan', 'Account length', 'Number vmail messages', 
           'Total day minutes', 'Total day calls', 'Total eve minutes', 'Total eve calls', 
           'Total night minutes', 'Total night calls', 'Total intl minutes', 'Total intl calls', 
           'Customer service calls']


def predict_churn(input_data):
    # Create a dataframe with the input data
    input_data = pd.DataFrame(input_data, columns=columns)

    # Encode the categorical variables
    label_encoder = LabelEncoder()
    input_data['International plan'] = label_encoder.fit_transform(input_data['International plan'])
    input_data['Voice mail plan'] = label_encoder.fit_transform(input_data['Voice mail plan'])

    input_df = pd.DataFrame([input_data], columns=columns)
    # Perform scaling on the numerical columns
    numerical_cols = ['Account length', 'Number vmail messages', 'Total day minutes', 'Total day calls',
                      'Total eve minutes', 'Total eve calls', 'Total night minutes', 'Total night calls',
                      'Total intl minutes', 'Total intl calls', 'Customer service calls']
    
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
    # Make predictions
    prediction = model_random.predict(input_df)
    return prediction

def main():
    import streamlit as st
    st.title('Telco Customer Churn Prediction')
    st.write('This app predicts whether a customer will likely churn or not.')
    st.write('The app uses a pre-trained Random Forest Classifier model.')

    # Collect the input data
    international_plan = st.selectbox('International plan', ('Yes', 'No'))
    voice_mail_plan = st.selectbox('Voice mail plan', ('Yes', 'No'))
    #account_length = st.number_input('Account length', min_value=0, max_value=100)
    account_length = st.slider('Account length', min_value=0, max_value=100)
    number_vmail_messages = st.number_input('Number vmail messages', min_value=0)
    total_day_minutes = st.number_input('Total day minutes', min_value=0)
    total_day_calls = st.number_input('Total day calls', min_value=0)
    total_eve_minutes = st.number_input('Total eve minutes', min_value=0)
    total_eve_calls = st.number_input('Total eve calls', min_value=0)
    total_night_minutes = st.number_input('Total night minutes', min_value=0)
    total_night_calls = st.number_input('Total night calls', min_value=0)
    total_intl_minutes = st.number_input('Total intl minutes', min_value=0)
    total_intl_calls = st.number_input('Total intl calls', min_value=0)
    customer_service_calls = st.number_input('Customer service calls', min_value=0)

    # Perform label encoding on the categorical columns
    label_encoder = LabelEncoder()
    international_plan = label_encoder.fit_transform([international_plan])[0]
    voice_mail_plan = label_encoder.fit_transform([voice_mail_plan])[0]

    input_df = pd.DataFrame({'International plan': [international_plan],
                  'Voice mail plan': [voice_mail_plan],
                  'Account length': [account_length],
                  'Number vmail messages': [number_vmail_messages],
                  'Total day minutes': [total_day_minutes],
                  'Total day calls': [total_day_calls],
                  'Total eve minutes': [total_eve_minutes],
                  'Total eve calls': [total_eve_calls],
                  'Total night minutes': [total_night_minutes],
                  'Total night calls': [total_night_calls],
                  'Total intl minutes': [total_intl_minutes],
                  'Total intl calls': [total_intl_calls],
                  'Customer service calls': [customer_service_calls]})

    # Perform scaling on the numerical columns
    numerical_cols = ['Account length', 'Number vmail messages', 'Total day minutes', 'Total day calls',
                      'Total eve minutes', 'Total eve calls', 'Total night minutes', 'Total night calls',
                      'Total intl minutes', 'Total intl calls', 'Customer service calls']
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

    # Make predictions using the pre-trained model
    prediction = model_random.predict(input_df)
    churn_probability = model_random.predict_proba(input_df)[:, 1]

    # Predict churn based on user input
    #churn_probability = predict_churn(input_data)
    #churn_prediction=churn_probability[1]

    # Display the prediction and churn probability
    if prediction[0] == 1:
            st.write(f'Churn Prediction: {prediction[0]}')
    else:
            st.write(f'Customer Not Churn')
    st.write(f'Churn Probability: {churn_probability[0]:.0%}')

    # Display the footer
    #st.markdown(footer,unsafe_allow_html=True)
       
        
    # Display the prediction as a percentage
    #st.write(f'The predicted probability of churn is {churn_prediction:.0%}.')

#Run the Streamlit application
if __name__ == '__main__':
    main()