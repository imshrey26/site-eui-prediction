import numpy as np
import pandas as pd
import streamlit as st
import joblib
from prediction import *
import sklearn

model = joblib.load(r'model/rf_tunedfinal.joblib')

st.set_page_config(page_title="Site Energy Intensity Prediction",
                   page_icon="🏢⚡", layout="wide")


options_state_factor = ['State_1', 'State_2', 'State_4', 'State_6', 'State_8', 'State_10',
       'State_11']
options_building_class = ['Commercial', 'Residential']
options_facility_type = ['Grocery_store_or_food_market',
       'Warehouse_Distribution_or_Shipping_center',
       'Retail_Enclosed_mall', 'Education_Other_classroom',
       'Warehouse_Nonrefrigerated', 'Warehouse_Selfstorage',
       'Office_Uncategorized', 'Data_Center', 'Commercial_Other',
       'Mixed_Use_Predominantly_Commercial',
       'Office_Medical_non_diagnostic', 'Education_College_or_university',
       'Industrial', 'Laboratory',
       'Public_Assembly_Entertainment_culture',
       'Retail_Vehicle_dealership_showroom', 'Retail_Uncategorized',
       'Lodging_Hotel', 'Retail_Strip_shopping_mall',
       'Education_Uncategorized', 'Health_Care_Inpatient',
       'Public_Assembly_Drama_theater', 'Public_Assembly_Social_meeting',
       'Religious_worship', 'Mixed_Use_Commercial_and_Residential',
       'Office_Bank_or_other_financial', 'Parking_Garage',
       'Commercial_Unknown', 'Service_Vehicle_service_repair_shop',
       'Service_Drycleaning_or_Laundry', 'Public_Assembly_Recreation',
       'Service_Uncategorized', 'Warehouse_Refrigerated',
       'Food_Service_Uncategorized', 'Health_Care_Uncategorized',
       'Food_Service_Other', 'Public_Assembly_Movie_Theater',
       'Food_Service_Restaurant_or_cafeteria', 'Food_Sales',
       'Public_Assembly_Uncategorized', 'Nursing_Home',
       'Health_Care_Outpatient_Clinic', 'Education_Preschool_or_daycare',
       '5plus_Unit_Building', 'Multifamily_Uncategorized',
       'Lodging_Dormitory_or_fraternity_sorority',
       'Public_Assembly_Library', 'Public_Safety_Uncategorized',
       'Public_Safety_Fire_or_police_station', 'Office_Mixed_use',
       'Public_Assembly_Other', 'Public_Safety_Penitentiary',
       'Health_Care_Outpatient_Uncategorized', 'Lodging_Other',
       'Mixed_Use_Predominantly_Residential', 'Public_Safety_Courthouse','Public_Assembly_Stadium', 'Lodging_Uncategorized',
       '2to4_Unit_Building', 'Warehouse_Uncategorized']


st.markdown("<h1 style='text-align: center;'>Site Energy Intensity Prediction Application 🏢⚡</h1>", unsafe_allow_html=True)

def main():
     with st.form('prediction_form'):
        
        st.subheader("Enter the input for following features:")

        year_factor = st.slider('Year Factor: ',1.0,6.0,format='%d')
        ratings = st.slider("Pickup energy star rating: ", 0, 100, value=0, format="%f")
        elevation = st.number_input(label='Enter Elevation of Building',format='%.1f')

        cooling_days = st.number_input(label='Enter cooling degree days',format='%.1f')
        heating_days = st.number_input(label='Enter heating degree days',format='%.1f')
        precipitation_inches = st.number_input(label='Enter precipitation inched',format='%.1f')
        snowfall_inches = st.number_input(label='Enter Snowfall im inches',format='%.1f')
        snowdepth_inches = st.number_input(label='Enter Snowdepth in inches',format='%.1f')
        avg_temp = st.number_input(label='Enter Average temperature',format='%.1f')

        state_factor = st.selectbox("Select state factor: ", options=options_state_factor)
        facility_type = st.selectbox("Select facility type: ", options=options_facility_type)

        avg_min_temp_winter = st.number_input(label='Enter Average minimum temperature in winter',format='%.1f')
        avg_max_temp_winter = st.number_input(label='Enter Average maximum temperature in winter',format='%.1f')
        avg_temp_winter = st.number_input(label='Enter Average temperature in winter',format='%.1f')

        avg_min_temp_summer = st.number_input(label='Enter Average minimum temperature in summer',format='%.1f')
        avg_max_temp_summer = st.number_input(label='Enter Average maximum temperature in summer',format='%.1f')
        avg_temp_summer = st.number_input(label='Enter Average temperature in summer',format='%.1f')

        avg_days_below30F = st.number_input(label='Enter Average days below 30F',format='%.1f')
        avg_days_below80F = st.number_input(label='Enter Average days below 80F',format='%.1f')

        try:
            floor_energy_star_rating = ratings/elevation

        except:
            print("Error: Division by zero is not allowed")


        submit = st.form_submit_button("Predict")


     if submit:
         
       state_factor = encodeStateFactor(state_factor)
       facility_type = encodeFacilityType(facility_type)

       data = np.array([year_factor,ratings,elevation,cooling_days,heating_days,precipitation_inches,
                        snowfall_inches,snowdepth_inches,avg_temp,state_factor,facility_type,
                        avg_min_temp_winter,avg_max_temp_winter,avg_temp_winter,
                        avg_min_temp_summer,avg_max_temp_summer,avg_temp_summer,
                        avg_days_below30F,avg_days_below80F,floor_energy_star_rating]).reshape(1,-1)
    
       pred = get_prediction(data, model=model)
    
       st.write('Site\'s Energy Use Intensity is: {}'.format(pred[0]))

if __name__ == '__main__':
    main()