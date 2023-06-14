import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import xgboost
import joblib
from PIL import Image
import requests
from io import BytesIO

st.header('Real Estate Price Prediction')

# Image URL
image_url = 'https://raw.githubusercontent.com/dragan-serghei/real_estate_intelligence/main/house_vibrant_forest.jpeg'

# Fetch image from URL
response = requests.get(image_url)
image = Image.open(BytesIO(response.content))

# Display image
st.image(image, caption='Bing Image Creator | Image created with AI - Powered by DALL-E', use_column_width=True)

# Import data
real_estate_df = pd.read_csv("https://raw.githubusercontent.com/dragan-serghei/real_estate_intelligence/main/real_estate_data.csv")

# Loading the dictionary from the .npy file
le_dict = np.load('label_encoder_classes.npy', allow_pickle=True).item()

# Retrieving the LabelEncoder object for the 'partitioning' variable
le_partitioning = le_dict['partitioning']

# Retrieving the LabelEncoder object for the 'structura_rezistenta' variable
le_rezistenta = le_dict['structura_rezistenta']

# Retrieving the LabelEncoder object for the 'localitate' variable
le_localitate = le_dict['localitate']

# Retrieving the LabelEncoder object for the 'tip_imobil' variable
le_tip_imobil = le_dict['tip_imobil']

# Retrieving the LabelEncoder object for the 'zona' variable
le_zona = le_dict['zona']

# Load XGBRegressor model
xgb_model = XGBRegressor()
xgb_model = joblib.load("best_model.joblib.gz")

# Display 100 rows of the dataset if box is checked
if st.checkbox('Show Training Dataframe'):
    st.write(real_estate_df.head(100))

# Define the user interface
st.markdown('Select the information about your property:')

# Add city to drop-down list
localitate = real_estate_df['localitate'].unique()
# Select city from drop-down list
selected_localitate = st.selectbox("City:", localitate)

# Add district to drop-down list
zona = real_estate_df['zona'].unique()
# Select district from drop-down list
selected_district = st.selectbox("Select District:", zona)

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

# Select max floor of the building
select_max_floor = st.slider('What is the max floor of the building?', -1, 30, 1)

# Check if mandarda
attic = st.selectbox('Is it located in the attic?', ('Yes', 'No'))
if attic == 'Yes':
    attic = True
else:
    attic = False

# Select furnishing level
select_furnishing = st.slider('Furnishing Level:', 0, max(real_estate_df["furnishing"]), 1)

# Select nr of bathrooms
select_bathrooms = st.slider('Number of Bathrooms:', 0, 10, 1)

# Select nr of balconies
select_balconies = st.slider('Number of Balconies:', 0, 10, 1)

# Select nr of kitchens
select_kitchens = st.slider('Number of Kitchens:', 0, 10, 1)

# Select nr of rooms
select_rooms = st.slider('Number of Rooms:', 0, 20, 1)

# Select nr parking slots
select_parking_slots = st.slider('Number of Parking Slots:', 0, 5, 1)

# Select Structural resistance - radio buttons
left_column, right_column = st.columns(2)
with left_column:
    structural_resistance = st.radio(
        'Structural resistance:',
        np.unique(real_estate_df['structura_rezistenta']))

# Select surface m2
select_surface = st.slider("What's the Living Surface (sqm) of your property?", 0.0, 500.0, 1.0)

# Select property type
left_column, right_column = st.columns(2)
with left_column:
    property_type = st.radio(
        'Is it an Apartment or a House?',
        np.unique(real_estate_df['tip_imobil']))

# Check if under construction
under_construction = st.selectbox('Is the property still under Construction?', ('Yes', 'No'))
if under_construction == 'Yes':
    under_construction = True
else:
    under_construction = False

# Check if in project phase  
project_phase = st.selectbox('Is the property at the Project Phase?', ('Yes', 'No'))
if project_phase == 'Yes':
    project_phase = True
else:
    project_phase = False

# Check district heating
district_heating = st.selectbox('Does it have a Centralized Heating System?', ('Yes', 'No'))
if district_heating == 'Yes':
    district_heating = True
else:
    district_heating = False

# Check building heating
building_heating = st.selectbox('Does it have a Building Heating System?', ('Yes', 'No'))
if building_heating == 'Yes':
    building_heating = True
else:
    building_heating = False

# Check invidual heating
individual_heating = st.selectbox('Does it have an Individual Heating System?', ('Yes', 'No'))
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
days_since_listing = st.slider("How many days have passed since you're actively trying to sell the property?", 0, 2000, 1)


if st.button('Predict House Price'):
    inpt_partitioning = le_partitioning.transform([partitioning_type])[0]
    inpt_localitate = le_localitate.transform([selected_localitate])[0]
    inpt_structural_resistance = le_rezistenta.transform([structural_resistance])[0]
    inpt_property_type = le_tip_imobil.transform([property_type])[0]
    inpt_selected_district = le_zona.transform([selected_district])[0]

    inputs = np.expand_dims(
        [selected_construction_year, int(inpt_partitioning), select_comfort, select_floor_level,
         int(inpt_localitate), select_furnishing, select_bathrooms, select_balconies, select_kitchens,
         select_rooms, select_parking_slots, int(inpt_structural_resistance), select_surface, int(inpt_property_type),
         int(inpt_selected_district), under_construction, project_phase, select_max_floor, attic, district_heating,
         building_heating, individual_heating, underfloor_heating, days_since_listing], 0)
    
    prediction = xgb_model.predict(inputs)
    print("final pred", np.squeeze(prediction, -1))
    st.write(f"The price per m2 of your property is: {np.squeeze(prediction, -1):.0f} €")
    full_price = select_surface * prediction.item()
    st.write(f"The full price of your property is: {full_price:.0f} €")
