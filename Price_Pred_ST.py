import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import category_encoders as ce
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score


df = pd.read_csv('STdata.csv')
df = df.drop(columns=['Unnamed: 0'])
X = df.drop(columns=['Price(Lakhs)'])
y = df['Price(Lakhs)']
y_log = np.log(y)

encoder = ce.TargetEncoder(cols=['Brand(Model)'])
X = encoder.fit_transform(X, y_log)

X_train, X_test, y_train_log, y_test_log = train_test_split(X, y_log, test_size = 0.2, random_state = 42)

scaling_list = ['Driven(KM)', 'Max Power', 'Age(Months)', 'Area']
scaler = StandardScaler()
X_train[scaling_list] = scaler.fit_transform(X_train[scaling_list])
X_test[scaling_list] = scaler.transform(X_test[scaling_list])
    
ohe_en_cols = ['Fuel Type', 'Transmission', 'City']
ohencoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

X_train_encoded = ohencoder.fit_transform(X_train[ohe_en_cols])
X_test_encoded = ohencoder.transform(X_test[ohe_en_cols])

encoded_columns = ohencoder.get_feature_names_out(ohe_en_cols)

X_train_encoded_df = pd.DataFrame(X_train_encoded, columns=encoded_columns, index=X_train.index)
X_test_encoded_df = pd.DataFrame(X_test_encoded, columns=encoded_columns, index=X_test.index)

X_train = pd.concat([X_train.drop(ohe_en_cols, axis=1), X_train_encoded_df], axis=1)
X_test = pd.concat([X_test.drop(ohe_en_cols, axis=1), X_test_encoded_df], axis=1)

rfregressor = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rfregressor.fit(X_train, y_train_log)


st.sidebar.markdown('<h1 class="sidebar-heading">Navigation Bar</h1>', unsafe_allow_html=True)
options = st.sidebar.radio('**select option**', ['Home', 'View Data', 'Predict', 'Contact Us'])

if options == 'Home':

    def add_bg_from_local():
        st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("https://miro.medium.com/v2/resize:fit:648/1*kQBj7l-Y1WPZfX9nKIYL1Q.jpeg");
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
        )

    add_bg_from_local()
    
    st.title('About Us')
    st.markdown('''
    **_:red-background[Our aim is to enhance the customer experience and streamline the pricing process of the cars by leveraging available used Car Data. We         create an accurate and user-friendly streamlit tool that predicts the prices of used cars based on various features. This interactive web application is        for both customers and sales representatives to use seamlessly.]_**
    ''')
if options == 'View Data':

    st.title('Available Car Information')
    st.write(df)
    
if options == 'Contact Us':
    
    st.title('Contact Us')
    name = st.text_input('Name:', value = 'Vijaya Raghava')
    email = st.text_input('Email:', value = 'vijay139@gmail.com')
    
if options == 'Predict':

    # def add_bg_from_local():
    #     st.markdown(
    #     f"""
    #     <style>
    #     .stApp {{
    #         background-image: url("https://www.sansoneauto.com/blogs/3504/wp-content/uploads/2022/11/calculating-car-payment.png");
    #         background-size: cover;
    #     }}
    #     </style>
    #     """,
    #     unsafe_allow_html=True
    #     )

    # add_bg_from_local()

    brand_list = df['Brand(Model)'].drop_duplicates().to_list()
    fuel_list = df['Fuel Type'].drop_duplicates().to_list()
    trans_list = df['Transmission'].drop_duplicates().to_list()
    city_list = df['City'].drop_duplicates().to_list()

    model = st.selectbox('Brand(Model):', brand_list)
    transmission = st.radio('Transmission:', trans_list)
    fuel = st.radio('Fuel Type:', fuel_list)
    age = st.number_input('Age(Years):', min_value=0, max_value=22, value=1, step=1)
    power = st.slider('Max Power(HP):', min_value=20, max_value=600, value=20)
    gear = st.number_input('Gear Box:', min_value=4, max_value=10, value=4, step=1)
    driven = st.slider('Driven(KM):', min_value=0, max_value=5500000, value=10)
    area = st.slider('Area(l*b):', min_value=8000, max_value=15000, value=10000)
    city = st.selectbox('City:', city_list)


    data = {'Driven(KM)':[driven], 'Brand(Model)':[model], 'Max Power':[power], 'Gear Box':[gear], 'Age(Months)':[age], 'Fuel Type':[fuel],  
           'Transmission':[transmission], 'City':[city], 'Area':[area]}
    X_test_new = pd.DataFrame(data)
    
    X_test_new = encoder.transform(X_test_new)

    scaling_list_new = ['Driven(KM)', 'Max Power', 'Age(Months)', 'Area']
    X_test_new[scaling_list_new] = scaler.transform(X_test_new[scaling_list_new])

    ohe_en_cols_new = ['Fuel Type', 'Transmission', 'City']

    X_test_encoded_new = ohencoder.transform(X_test_new[ohe_en_cols_new])
    encoded_columns_new = ohencoder.get_feature_names_out(ohe_en_cols_new)
    X_test_new_encoded_df = pd.DataFrame(X_test_encoded_new, columns=encoded_columns_new, index=X_test_new.index)
    X_test_new = pd.concat([X_test_new.drop(ohe_en_cols_new, axis=1), X_test_new_encoded_df], axis=1)


    if st.button('Predict'):
        y_pred_log = rfregressor.predict(X_test_new)
        y_pred = np.exp(y_pred_log)

        st.write(f"The predicted price of the car: Rs {y_pred*100000}")




