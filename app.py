import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
import streamlit as st

with open('pipeline.pkl','rb') as file1:
    pipeline = pickle.load(file1)

func_imp = tf.keras.models.load_model('model functional improve.h5')

region_category = st.selectbox('Select Region : ',('Village', 'City', 'Town'))
membership_category = st.selectbox('Select Membership Category',
                                    ('No Membership',
                                     'Basic Membership', 
                                     'Platinum Membership', 
                                     'Gold Membership', 
                                     'Premium Membership', 
                                     'Silver Membership'))
joined_through_referral = st.radio('Joined Through Referral? ',('Yes','No'))
preferred_offer_types = st.selectbox('Select Preferred Offer : ',('Credit/Debit Card Offers',
                                       'Gift Vouchers/Coupons',
                                       'Without Offers'))
medium_of_operation = st.selectbox('Select Medium of Operation : ',('Desktop', 'Smartphone', 'Both'))
days_since_last_login = st.number_input('Input Days Since Login : ',0,26)
avg_time_spent = st.number_input('Input Average Time Spent : ',0.0,3235.5785210942604)
avg_transaction_value = st.number_input('Input Average Transaction Value : ',800.46,99914.05)
avg_frequency_login_days = st.number_input('Input Average Frequency Login Days : ',0.0,73.06199459430009)
points_in_wallet = st.number_input('Input Points in Wallet : ',0.0,2069.069760814851)
used_special_discount = st.radio('Used Special Discount? ',('Yes','No'))
offer_application_preference = st.radio('Offer Application Preference : ',('Yes','No'))
feedback = st.selectbox('Select Feedback : ',('Too many ads', 'No reason specified', 'Reasonable Price',
                          'Quality Customer Care', 'Poor Website', 'Poor Customer Service',
                          'Poor Product Quality', 'User Friendly Website', 'Products always in Stock'))

if st.button('Predict'):
    data_inf = pd.DataFrame({'region_category' : region_category,
                             'membership_category' : membership_category,
                             'joined_through_referral' : joined_through_referral,
                             'preferred_offer_types' : preferred_offer_types,
                             'medium_of_operation' : medium_of_operation,
                             'days_since_last_login' : days_since_last_login,
                             'avg_time_spent' : avg_time_spent,
                             'avg_transaction_value' : avg_transaction_value,
                             'avg_frequency_login_days' : avg_frequency_login_days,
                             'points_in_wallet' : points_in_wallet,
                             'used_special_discount' : used_special_discount,
                             'offer_application_preference' : offer_application_preference,
                             'feedback' : feedback}, index=[0])
    hasil = 'Not Churn' if np.round(func_imp.predict(pipeline.transform(data_inf))) == 0 else 'Churn'
    st.header(f'Prediksi = {hasil}')