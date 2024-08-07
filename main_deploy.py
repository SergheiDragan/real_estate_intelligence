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
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

st.markdown("<h2 style='text-align: center;'>Descoperă valoarea reală a casei tale!</h2>", unsafe_allow_html=True)

description = '''
Acesta este un instrument de estimare a prețului pentru proprietățile din România bazat pe un model de învățare automată antrenat pe un set de date ce conține aproximativ 70.000 de proprietăți din 9 cele mai mari orașe.
'''
st.markdown(description)

# Image URL
image_url = 'https://raw.githubusercontent.com/SergheiDragan/real_estate_intelligence/main/Modern_house_in_the_forest.png'

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

# Define the user interface
st.markdown('Select the information about your property:')

# Add city to drop-down list
localitate = real_estate_df['localitate'].unique()
# Select city from drop-down list
selected_localitate = st.selectbox("City:", localitate)

# Filter regions based on the selected city
filtered_regions = real_estate_df[real_estate_df['localitate'] == selected_localitate]['zona'].unique()
# Sort regions in ascending order
filtered_regions = np.sort(filtered_regions)
# Select region from the filtered options
selected_district = st.selectbox("Select District:", filtered_regions)

# Add construction year to drop-down list
construction_year = real_estate_df['construction_year'].unique()
# Sort construction year in descending order
construction_year = np.sort(construction_year)[::-1]
# Select construction year from drop-down list
selected_construction_year = st.selectbox("Construction year:", construction_year)

# Select apartment partitioning type - radio buttons
left_column, right_column = st.columns(2)
with left_column:
    partitioning_type = st.radio(
        'Type of Partitioning:',
        np.unique(real_estate_df['partitioning']))

# Select comfort level slider
select_comfort = st.slider('Comfort Level (0 -> highest comfort to 3 -> lowest comfort):', 0, max(real_estate_df["comfort"]), 1)

# Select floor level
select_floor_level = st.slider('Floor Level:', -1, max(real_estate_df["floor_level"]), 1)

# Select max floor of the building
select_max_floor = st.slider('What is the max floor of the building?', 0, 30, 1)

# Check if mandarda
attic = st.selectbox('Is it located in the attic?', ('Yes', 'No'))
if attic == 'Yes':
    attic = True
else:
    attic = False

# Select furnishing level
select_furnishing = st.slider('Furnishing Level (0 -> unfurnished to 3 -> luxury furnishing):', 0, max(real_estate_df["furnishing"]), 1)

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
select_surface = st.slider("What's the Living Surface (sqm) of your property?", 0, 500, 1)

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

# Initialize heating systems to False
district_heating = False
building_heating = False
individual_heating = False

# Heating system selection
heating_options = ('Centralized Heating', 'Building Heating', 'Individual Heating')
selected_heating = st.selectbox('What type of heating system does the property have?', heating_options)

# Update the selected heating system to True
if selected_heating == 'Centralized Heating':
    district_heating = True
elif selected_heating == 'Building Heating':
    building_heating = True
elif selected_heating == 'Individual Heating':
    individual_heating = True

# Check Underfloor heating
underfloor_heating = st.selectbox('Underfloor Heating?', ('Yes', 'No'))
if underfloor_heating == 'Yes':
    underfloor_heating = True
else:
    underfloor_heating = False

# Select days since listing
days_since_listing = st.slider("How many days have passed since you're actively trying to sell the property?", 0, 2000, 1)

# Calculate the range for surface m2 to be used for plotting the histogram
surface_min = (select_surface // 10) * 10
surface_max = surface_min + 9

# The range of years to be used for plotting the histogram
previous_years = selected_construction_year - 2

# Filter the DataFrame based on user-selected attributes to be used for the Histogram
filtered_df = real_estate_df[
    (real_estate_df['localitate'] == selected_localitate) &
    (real_estate_df['zona'] == selected_district) &
    (real_estate_df['construction_year'].between(previous_years, selected_construction_year)) &
    (real_estate_df['rooms'] == select_rooms) &
    (real_estate_df['useful_surface'].between(surface_min, surface_max))
]

# Apply a custom color to the histogram
color = '#FFA726'

# Apply the "seaborn" theme to the plot
sns.set_theme()

# Add a note for the user
note = f"The histogram below shows the distribution of properties' prices based on the selected attributes:\n" \
       f"- City: <b>{selected_localitate}</b>\n" \
       f"- District: <b>{selected_district}</b>\n" \
       f"- Construction Year: <b>{previous_years}-{selected_construction_year}</b>\n" \
       f"- Number of Rooms: <b>{select_rooms}</b>\n" \
       f"- Living Surface: <b>{surface_min}-{surface_max} m2</b>"

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
    price_per_m2 = np.squeeze(prediction, -1)
    formatted_price_per_m2 = "{:,.0f}".format(price_per_m2)
    
    st.write(f"The price per m2 of your property is: <span style='font-weight: bold; font-size: 20px'>{formatted_price_per_m2} €</span>", unsafe_allow_html=True)
    full_price = select_surface * prediction.item()
    formatted_full_price = "{:,.0f}".format(full_price)
    
    st.write(f"The full price of your property is: <span style='font-weight: bold; font-size: 20px'>{formatted_full_price} €</span>", unsafe_allow_html=True)

    # Check if there are properties that match the selected criteria
    num_properties = len(filtered_df)
    
    if num_properties < 5:
        st.write(f"Based on the selected criteria, there are less than 5 properties available. A histogram is generated only if the number of properties that satisfy the selected criteria meet this threshold.")
    else:    
        # Create the histogram using matplotlib
        fig, ax = plt.subplots(1, 1)
        fig.set_figheight(4) # Adjust the figure height
        ax.hist(filtered_df['price_EUR_sqm'], bins=10, color=color)
    
        # Set y-axis to integer values
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        # Add a blue line for the estimated price
        ax.axvline(price_per_m2, color='blue', linestyle='--', linewidth=2, label='Estimated Price')
        
        # Set labels and title
        fontsize = 9
        ax.set_xlabel('Property Price (EUR/m2)', fontsize=fontsize)
        ax.set_ylabel('# of Properties',fontsize=fontsize)
        ax.set_title('Histogram of Actual Property Prices (EUR/m2)', fontsize=10)
        ax.tick_params(axis='x', labelsize=9)
        ax.tick_params(axis='y', labelsize=9)
    
        # Note for the user to understand what the histogram shows 
        st.markdown(note, unsafe_allow_html=True)
    
        # Display the number of properties presented in the histogram
        num_properties = len(filtered_df)
        st.write(f"The number of properties that satisfy the above selected criteria is: {num_properties}")

        # Display the legend
        ax.legend(fontsize=8)
        # Display the histogram
        st.pyplot(fig)
