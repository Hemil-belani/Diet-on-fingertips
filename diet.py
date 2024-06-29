import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
import scipy.stats as stats
import pylab

# Load the data
data = pd.read_csv('recipes.csv')

# Define the maximum daily nutritional values
max_Calories = 2000
max_daily_fat = 100
max_daily_Saturatedfat = 13
max_daily_Cholesterol = 300
max_daily_Sodium = 2300
max_daily_Carbohydrate = 325
max_daily_Fiber = 40
max_daily_Sugar = 40
max_daily_Protein = 200
max_list = [max_Calories, max_daily_fat, max_daily_Saturatedfat, max_daily_Cholesterol, max_daily_Sodium, max_daily_Carbohydrate, max_daily_Fiber, max_daily_Sugar, max_daily_Protein]

# Data extraction and processing functions
def extract_data(dataframe, ingredient_filter, max_nutritional_values):
    extracted_data = dataframe.copy()
    for column, maximum in zip(extracted_data.columns[6:15], max_nutritional_values):
        extracted_data = extracted_data[extracted_data[column] < maximum]
    if ingredient_filter:
        for ingredient in ingredient_filter:
            extracted_data = extracted_data[extracted_data['RecipeIngredientParts'].str.contains(ingredient, regex=False)]
    return extracted_data

def scaling(dataframe):
    scaler = StandardScaler()
    prep_data = scaler.fit_transform(dataframe.iloc[:, 6:15].to_numpy())
    return prep_data, scaler

def nn_predictor(prep_data):
    neigh = NearestNeighbors(metric='cosine', algorithm='brute')
    neigh.fit(prep_data)
    return neigh

def build_pipeline(neigh, scaler, params):
    transformer = FunctionTransformer(neigh.kneighbors, kw_args=params)
    pipeline = Pipeline([('std_scaler', scaler), ('NN', transformer)])
    return pipeline

def apply_pipeline(pipeline, _input, extracted_data):
    return extracted_data.iloc[pipeline.transform(_input)[0]]

def recommend(dataframe, _input, max_nutritional_values, ingredient_filter=None, params={'return_distance': False}):
    extracted_data = extract_data(dataframe, ingredient_filter, max_nutritional_values)
    prep_data, scaler = scaling(extracted_data)
    neigh = nn_predictor(prep_data)
    pipeline = build_pipeline(neigh, scaler, params)
    return apply_pipeline(pipeline, _input, extracted_data)

# Streamlit app
st.title('Diet Recipe Recommendation')

age = st.number_input('Enter your age', min_value=0, max_value=100, value=25)
height = st.number_input('Enter your height (in cm)', min_value=50, max_value=250, value=170)
weight = st.number_input('Enter your weight (in kg)', min_value=20, max_value=200, value=70)
gender = st.selectbox('Select your gender', ['Male', 'Female'])

# Calculate BMI
bmi = weight / (height / 100) ** 2

st.write(f'Your BMI is {bmi:.2f}')

# Placeholder for user nutritional needs (you might want to customize these values)
user_input = np.array([[max_Calories, max_daily_fat, max_daily_Saturatedfat, max_daily_Cholesterol, max_daily_Sodium, max_daily_Carbohydrate, max_daily_Fiber, max_daily_Sugar, max_daily_Protein]])

if st.button('Recommend a Diet Recipe'):
    recommendations = recommend(data, user_input, max_list)
    st.write(recommendations)

    # Visualize the histogram of calories
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.title('Frequency Histogram')
    plt.ylabel('Frequency')
    plt.xlabel('Bins Center')
    ax.hist(data.Calories.to_numpy(), bins=[0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 5000], linewidth=0.5, edgecolor="white")
    st.pyplot(fig)

    # Q-Q plot for calories
    stats.probplot(data.Calories.to_numpy(), dist="norm", plot=pylab)
    st.pyplot(pylab)

# Run the Streamlit app
if __name__ == "__main__":
    st.write("Run the streamlit app using the command: streamlit run app.py")
