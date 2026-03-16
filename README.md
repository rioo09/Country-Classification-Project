# 🌍 Country Development Cluster Prediction

A Machine Learning project that analyzes global development indicators and groups countries into development categories using clustering techniques.
The project includes an interactive web application built with Streamlit for predicting the development cluster of a country based on socio-economic indicators.

URL : https://country-classification-project-mxns2yb6emum74pbyvrgbk.streamlit.app

---

## 🚀 Project Overview

Countries differ widely in terms of economic strength, health infrastructure, technology adoption, and population structure.
This project applies **unsupervised machine learning** to identify patterns in these indicators and group countries into meaningful development clusters.

The model analyzes multiple global development metrics such as GDP, health expenditure, population distribution, technology usage, and environmental indicators.

The results help categorize countries into different development levels such as:

* Developed Countries
* Emerging Economies
* Developing Countries
* Underdeveloped Countries

---

## 🧠 Machine Learning Workflow

The pipeline used in this project follows the standard data science workflow:

Dataset
→ Data Cleaning
→ Feature Scaling
→ Dimensionality Reduction (PCA)
→ Clustering Model (Gaussian Mixture Model)
→ Cluster Prediction

---

## 📊 Features Used

The model uses multiple development indicators including:

* Birth Rate
* CO₂ Emissions
* Days to Start Business
* Energy Usage
* GDP
* Health Expenditure (% GDP)
* Health Expenditure per Capita
* Infant Mortality Rate
* Internet Usage
* Lending Interest
* Life Expectancy (Male & Female)
* Mobile Phone Usage
* Population Age Distribution
* Population Urban
* Tourism Inbound & Outbound

These indicators represent **economic, demographic, environmental, and technological development**.

---

## 🤖 Model Used

This project uses the **Gaussian Mixture Model (GMM)** for clustering.

GMM is a probabilistic clustering algorithm that assumes the data is generated from a mixture of several Gaussian distributions and assigns probabilities for cluster membership.

Additionally, **Principal Component Analysis (PCA)** is applied to reduce dimensionality before clustering.

---

## 🖥 Interactive Web Application

The project includes an interactive dashboard built with **Streamlit**.

Users can:

1. Enter country development indicators
2. Run the clustering model
3. Predict the development category of the country

The interface includes:

* Modern UI with background visuals
* Glass-style input fields
* Cluster prediction display

---

## 📂 Project Structure

```
Country-Classification-Project
│
├── app.py
├── P-651.ipynb
├── gmm_model.pkl
├── scaler.pkl
├── pca.pkl
├── final_df_with_clusters.csv
├── requirements.txt
└── 3d-rendering-planet-earth.jpg
```

---

## ⚙️ Installation

Clone the repository:

```
git clone https://github.com/rioo09/Country-Classification-Project.git
```

Navigate to the project folder:

```
cd Country-Classification-Project
```

Install dependencies:

```
pip install -r requirements.txt
```

Run the Streamlit application:

```
streamlit run app.py
```

---

## 🌐 Live Demo

The application can be deployed using **Streamlit Community Cloud**.

After deployment, the app can be accessed directly via a public URL.
URL : https://country-classification-project-mxns2yb6emum74pbyvrgbk.streamlit.app

---

## 🛠 Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Joblib
* Streamlit

---

## 📈 Future Improvements

Possible enhancements for the project:

* Country dropdown with automatic indicator loading
* Cluster visualization using PCA plots
* Interactive world map showing cluster distribution
* Model comparison with K-Means clustering

---

## 👤 Author

**Rio**

Data Science & Machine Learning Enthusiast

GitHub:
https://github.com/rioo09

---

⭐ If you found this project useful, consider giving it a star!
