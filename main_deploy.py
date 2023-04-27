import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import xgboost
import joblib

st.header('Real Estate Price Prediction')

# Import data
real_estate_df = pd.read_csv('real_estate_data.csv') # put it on github and paste the link to csv

# Load LabelEncoder class
le = LabelEncoder()
le.classes_ = np.load('label_encoder_classes.npy', allow_pickle=True)

# Load XGBRegressor model
xgb_model = XGBRegressor()
xgb_model = joblib.load("best_model.joblib.gz")



# Display 100 rows of the dataset if box is checked
if st.checkbox('Show Training Dataframe'):
    real_estate_df.head(100)

# Define the user interface
st.markdown('Select the information about your property:')

# Add construction year to drop-down list
construction_year = real_estate_df['construction_year'].unique()
# Select construction year from drop-down list
selected_construction_year = st.selectbox("Construction year:", construction_year)

# Select apartment partitioning type - radio buttons
left_column, right_column = st.columns(2)
with left_column:
    partitioning_type = st.radio(
        'Type of Partitioning:',
        np.unique(real_estate_df['partitioning']))

# Select comfort level slider
select_comfort = st.slider('Comfor Level:', 0, max(real_estate_df["comfort"]), 1)

# Select floor level
select_floor_level = st.slider('Floor Level:', 0, max(real_estate_df["floor_level"]), 1)

# Add city to drop-down list
localitate = real_estate_df['localitate'].unique()
# Select city from drop-down list
selected_localitate = st.selectbox("City:", localitate)

# Select furnishing level
select_furnishing = st.slider('Furnishing Level:', 0, max(real_estate_df["furnishing"]), 1)

# Select nr of bathrooms
select_bathrooms = st.slider('# of Bathrooms:', 0, max(real_estate_df["nr_bai"]), 1)

# Select nr of balconies
select_balconies = st.slider('# of Balconies:', 0, max(real_estate_df["nr_balcoane"]), 1)

# Select nr of kitchens
select_kitchens = st.slider('# of Kitchens:', 0, max(real_estate_df["nr_of_kitchens"]), 1)

# Select nr of rooms
select_rooms = st.slider('# of Rooms:', 0, max(real_estate_df["rooms"]), 1)

# Select nr parking slots
select_parking_slots = st.slider('# of Parking Slots:', 0, max(real_estate_df["nr_locuri_parcare"]), 1)

# Select Structural resistance - radio buttons
left_column, right_column = st.columns(2)
with left_column:
    structural_resistance = st.radio(
        'Structural resistance:',
        np.unique(real_estate_df['structura_rezistenta']))

# Select surface m2
select_surface = st.slider('Surface (sqm):', 0.0, max(real_estate_df["useful_surface"]), 1.0)

# Select property type
left_column, right_column = st.columns(2)
with left_column:
    property_type = st.radio(
        'Apartment or House?',
        np.unique(real_estate_df['tip_imobil']))

# Add district to drop-down list
zona = real_estate_df['zona'].unique()
# Select district from drop-down list
selected_district = st.selectbox("District:", zona)

# Check if under construction
under_construction = st.selectbox('Under Construction?', ('Yes', 'No'))
if under_construction == 'Yes':
    under_construction = True
else:
    under_construction = False

# Check if in project phase  
project_phase = st.selectbox('Project Phase?', ('Yes', 'No'))
if project_phase == 'Yes':
    project_phase = True
else:
    project_phase = False

# Select max floor of the building
select_max_floor = st.slider('What is the max floor of the building?', -1, max(real_estate_df["max_floor"]), 1)

# Check if mandarda
attic = st.selectbox('Is attic?', ('Yes', 'No'))
if attic == 'Yes':
    attic = True
else:
    attic = False

# Check district heating
district_heating = st.selectbox('Centralized Heating System?', ('Yes', 'No'))
if district_heating == 'Yes':
    district_heating = True
else:
    district_heating = False

# Check building heating
building_heating = st.selectbox('Building Heating System?', ('Yes', 'No'))
if building_heating == 'Yes':
    building_heating = True
else:
    building_heating = False

# Check invidual heating
individual_heating = st.selectbox('Individual Heating System?', ('Yes', 'No'))
if individual_heating == 'Yes':
    individual_heating = True
else:
    individual_heating = False

# Check Underfloor heating
underfloor_heating = st.selectbox('Underfloor Heating?', ('Yes', 'No'))
if underfloor_heating == 'Yes':
    underfloor_heating = True
else:
    underfloor_heating = False

# Select days since listing
days_since_listing = st.slider('How many days have passed since listing the property?', 0, max(real_estate_df["days_since_listing"]), 1)


if st.button('Predict Price per m2'):
    inpt_partitioning = le.transform(np.expand_dims(partitioning_type, -1))
    inpt_localitate = le.transform(np.expand_dims(selected_localitate, -1))
    inpt_structural_resistance = le.transform(np.expand_dims(structural_resistance, -1))
    inpt_property_type = le.transform(np.expand_dims(property_type, -1))
    inpt_selected_district = le.transform(np.expand_dims(selected_district, -1))

    inputs = np.expand_dims(
        [selected_construction_year, int(inpt_partitioning), select_comfort, select_floor_level,
         int(inpt_localitate), select_furnishing, select_bathrooms, select_balconies, select_kitchens,
         select_rooms, select_parking_slots, int(inpt_structural_resistance), select_surface, int(inpt_property_type),
         int(inpt_selected_district), under_construction, project_phase, select_max_floor, attic, district_heating,
         building_heating, individual_heating, underfloor_heating, days_since_listing], 0)
    
    prediction = xgb_model.predict(inputs)
    print("final pred", np.squeeze(prediction, -1))
    st.write(f"Your aparment fair value per m2 is: {np.squeeze(prediction, -1):.2f} €")