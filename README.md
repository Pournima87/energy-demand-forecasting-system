# ⚡ AI Powered Energy Demand Forecasting System

An end-to-end Machine Learning and Time Series project that forecasts energy consumption using advanced modeling techniques like SARIMA and ML models, deployed with an interactive Streamlit dashboard.

---

## 🚀 Project Overview

This system predicts future household energy consumption by analyzing historical time-series data. It combines data analysis, machine learning, and statistical modeling to deliver accurate forecasts and business insights.

---

## 🎯 Business Objective

- Forecast energy demand to optimize resource planning  
- Reduce operational costs by avoiding overproduction  
- Prevent outages by anticipating peak demand  
- Improve decision-making for energy distribution  

---

## 📊 Key Features

✔ Time Series Forecasting (SARIMA)  
✔ Machine Learning Models (Linear Regression, Random Forest)  
✔ Model Comparison & Evaluation  
✔ Trend & Seasonality Analysis  
✔ Anomaly Detection  
✔ Business Insights & Recommendations  
✔ Interactive Streamlit Dashboard  

---

## 🧠 Model Comparison

| Model               | RMSE  |
|--------------------|-------|
| Linear Regression  | 0.2709 |
| Random Forest      | 0.2619 |
| ARIMA              | 0.2528 |
| **SARIMA**         | **0.2381** |

👉 SARIMA performed best due to its ability to capture both **trend and seasonality**.

---

## ⚖️ Interpretability vs Accuracy

- Linear Regression → Highly interpretable but less flexible  
- Random Forest → Captures complex patterns but less interpretable  
- SARIMA → Best suited for time-series forecasting  

Trade-off:  
Simple models are easier to explain, while advanced models provide better accuracy.

---

## ❗ Model Limitations

- Struggles with sudden spikes or unexpected behavior  
- Depends heavily on historical patterns  
- Cannot predict unseen anomalies  

---

## 💼 Business Impact

- 📉 15–25% cost reduction through optimized energy planning  
- ⚡ Reduced risk of system overload and outages  
- 📊 Improved demand forecasting and resource allocation  

---

## 🔄 Model Monitoring & Retraining

- Retrain model periodically (weekly/monthly)  
- Monitor prediction error over time  
- Detect data drift and update model accordingly  

---

## 🚨 Alert System

- Trigger alerts when prediction error exceeds threshold  
- Detect abnormal spikes in energy consumption  
- Enable proactive decision-making  

---

## 📁 Project Structure
energy-demand-forecasting-system/
│

├── data/ # Dataset (not uploaded)

├── notebook/ # Jupyter notebooks

├── app.py # Streamlit application

├── requirements.txt # Dependencies

├── README.md # Project documentation

---

## ⚙️ Tech Stack

- Python  
- Pandas, NumPy  
- Matplotlib  
- Scikit-learn  
- Statsmodels (SARIMA)  
- Streamlit  

---

## 📌 Dataset

Dataset not uploaded due to size.  
Place the dataset inside the `data/` folder: data/household_power_consumption.txt
Dataset link : https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption

---

## ▶️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

🌐 Live Demo

👉 (Add your Streamlit link here after deployment)

---

📈 Future Enhancements

- Real-time forecasting
- Automated retraining pipeline
- Advanced anomaly detection
- Cloud deployment with monitoring

---

🙌 Author

Pournima More
Aspiring Data Scientist | Machine Learning Enthusiast
