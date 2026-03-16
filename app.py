import streamlit as st
import pandas as pd
import numpy as np
import joblib

# load trained objects
model = joblib.load("gmm_model.pkl")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")

st.set_page_config(page_title="Country Cluster Predictor", layout="wide")

# custom css
st.markdown("""
<style>

.stApp {
background-image: url("https://github.com/rioo09/Country-Classification-Project/blob/main/3d-rendering-planet-earth.jpg?raw=true");
background-size: cover;
background-attachment: fixed;
}

h1 {
color: white;
text-align: center;
font-size: 50px;
}

.block-container {
background: rgba(0,0,0,0.65);
padding: 2rem;
border-radius: 15px;
}

/* Glass input fields */
.stNumberInput input {
background: rgba(255,255,255,0.08) !important;
backdrop-filter: blur(8px);
border: 1px solid rgba(255,255,255,0.25) !important;
border-radius: 10px !important;
color: white !important;
}

.stNumberInput button {
background: rgba(255,255,255,0.15) !important;
color: white !important;
}

label {
color: white !important;
}

.stButton>button {
background: linear-gradient(90deg,#00c6ff,#0072ff);
color: white;
font-size: 20px;
border-radius: 10px;
padding: 10px 25px;
}

.result-box {
background: linear-gradient(135deg,#00c6ff,#0072ff);
padding: 20px;
border-radius: 12px;
text-align: center;
font-size: 28px;
color: white;
margin-top: 20px;
}
            
.glass-header {
background: rgba(255,255,255,0.08);
backdrop-filter: blur(10px);
-webkit-backdrop-filter: blur(10px);
border: 1px solid rgba(255,255,255,0.2);
border-radius: 15px;
padding: 20px;
text-align: center;
font-size: 42px;
font-weight: 600;
color: white;
margin-bottom: 30px;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="glass-header">
Country Development Cluster Prediction
</div>
""", unsafe_allow_html=True)

# load models
model = joblib.load("gmm_model.pkl")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")

columns = [
'Birth Rate','CO2 Emissions','Days to Start Business','Energy Usage',
'GDP','Health Exp % GDP','Health Exp/Capita','Infant Mortality Rate',
'Internet Usage','Lending Interest','Life Expectancy Female',
'Life Expectancy Male','Mobile Phone Usage','Population 0-14',
'Population 15-64','Population 65+','Population Total',
'Population Urban','Tourism Inbound','Tourism Outbound'
]

# columns layout
col1, col2, col3 = st.columns(3)

with col1:
    birth_rate = st.number_input("Birth Rate")
    co2 = st.number_input("CO2 Emissions")
    days_business = st.number_input("Days to Start Business")
    energy = st.number_input("Energy Usage")
    gdp = st.number_input("GDP")
    health_gdp = st.number_input("Health Exp % GDP")
    health_capita = st.number_input("Health Exp/Capita")

with col2:
    infant = st.number_input("Infant Mortality Rate")
    internet = st.number_input("Internet Usage")
    lending = st.number_input("Lending Interest")
    life_f = st.number_input("Life Expectancy Female")
    life_m = st.number_input("Life Expectancy Male")
    mobile = st.number_input("Mobile Phone Usage")
    pop_14 = st.number_input("Population 0-14")

with col3:
    pop_64 = st.number_input("Population 15-64")
    pop_65 = st.number_input("Population 65+")
    pop_total = st.number_input("Population Total")
    pop_urban = st.number_input("Population Urban")
    tour_in = st.number_input("Tourism Inbound")
    tour_out = st.number_input("Tourism Outbound")



if st.button("Predict Cluster"):

    data = [[birth_rate, co2, days_business, energy, gdp,
             health_gdp, health_capita, infant, internet, lending,
             life_f, life_m, mobile,
             pop_14, pop_64, pop_65,
             pop_total, pop_urban, tour_in, tour_out]]
    columns = ['Birth Rate','CO2 Emissions','Days to Start Business','Energy Usage','GDP','Health Exp % GDP','Health Exp/Capita','Infant Mortality Rate','Internet Usage','Lending Interest','Life Expectancy Female','Life Expectancy Male','Mobile Phone Usage','Population 0-14','Population 15-64','Population 65+','Population Total','Population Urban','Tourism Inbound','Tourism Outbound']

    df = pd.DataFrame(data, columns=columns)

    # Step 1: Scale
    scaled = scaler.transform(df)

    # Step 2: PCA
    pca_data = pca.transform(scaled)
    pca_df = pd.DataFrame(pca_data).head()  # Check PCA output
    final_df = pca_df.iloc[:,:6]


    # Step 3: Predict
    cluster = model.predict(final_df)
    cluster_names = {
    0: "Developed Country",
    1: "Emerging Economy",
    2: "Developing Country",
    3: "Underdeveloped Country"
     }

    result = cluster_names[int(cluster[0])]
    st.markdown(f"""
    <div class="result-box">
    Predicted Cluster : {result}
    </div>
    """, unsafe_allow_html=True)