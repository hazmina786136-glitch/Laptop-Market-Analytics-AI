# 💻 Laptop Market Analytics & AI Price Predictor

> **Data Analytics with AI Academic Internship 2026**  
> A complete Streamlit-based data analytics dashboard with an AI-powered laptop price prediction system.

---

## 🚀 Project Overview

The **Laptop Market Analytics & AI Price Predictor** is an end-to-end data analytics and machine learning project developed to analyze the laptop market and predict laptop prices based on important product specifications.

The project combines:

- 📊 Data Analytics
- 🤖 Machine Learning
- 📈 Interactive Visualizations
- 💰 Price Prediction
- 🎯 Business Insights
- 🖥️ Streamlit Dashboard

The dashboard helps users understand laptop market patterns such as **brand pricing, processor distribution, RAM, storage, customer segments, regional demand, and price relationships**.

---

## 🎯 Project Objective

The main objective of this project is to:

1. Analyze laptop market data.
2. Identify important pricing patterns.
3. Compare different laptop brands and specifications.
4. Understand customer segments and market trends.
5. Build an AI-based laptop price prediction model.
6. Generate actionable business insights.
7. Present all findings through an interactive dashboard.

---

## 💡 Business Problem

The laptop market contains a large number of products with different:

- Brands
- Processors
- RAM configurations
- Storage types
- Storage capacities
- GPUs
- Screen sizes
- Operating systems
- Customer segments
- Regions

Because of these differences, estimating an appropriate laptop price manually can be difficult.

This project uses **data analytics and machine learning** to analyze these factors and estimate the expected price of a laptop.

---

# 📊 Dashboard Features

## 1️⃣ Executive Overview

The dashboard provides important market KPIs including:

- 💰 Average Laptop Price
- 🏷️ Number of Brands
- 🥇 Leading Brand
- 💾 SSD Adoption
- 📦 Product Distribution
- 🌍 Regional Information

These KPIs provide a quick overview of the laptop market.

---

## 2️⃣ Market Analytics

The Market Analytics section analyzes:

### 🏷️ Brand Analysis

Compare laptop prices across different brands and identify major market players.

### ⚙️ Processor Analysis

Analyze the distribution of:

- Intel Core i3
- Intel Core i5
- Intel Core i7
- AMD Ryzen 3
- AMD Ryzen 5
- AMD Ryzen 7
- Apple M-Series

### 💾 Storage Analysis

Compare:

- SSD
- HDD

and understand storage preferences in the dataset.

### 👥 Customer Segment Analysis

Analyze laptop demand across different customer groups such as:

- Students
- Professionals
- Gamers
- Business Users
- Creators

---

# 📈 Pricing Analytics

The project analyzes the relationship between laptop specifications and price.

### RAM vs Price

Understand how laptop RAM capacity affects average laptop prices.

### Storage vs Price

Analyze how storage capacity is related to laptop pricing.

### Brand × Processor Analysis

Compare pricing patterns between different brands and processor categories.

---

# 🤖 AI Laptop Price Predictor

One of the major features of this project is an **AI-powered price prediction system**.

The system uses a:

> **Random Forest Regressor**

to estimate the expected laptop price.

### Input Features

Users can enter specifications such as:

- Brand
- Processor
- RAM
- Storage Type
- Storage Capacity
- GPU
- Screen Size
- Operating System
- Weight
- Battery Hours
- Warranty
- Customer Segment
- Region

The machine learning model then predicts the estimated laptop price.

---

# 🧠 Machine Learning Workflow

The machine learning pipeline follows these steps:

```text
Raw Dataset
     ↓
Data Cleaning
     ↓
Feature Selection
     ↓
Categorical Encoding
     ↓
Train-Test Split
     ↓
Random Forest Regressor
     ↓
Model Evaluation
     ↓
Price Prediction
```

---

# 📐 Model Evaluation

The model is evaluated using commonly used regression metrics:

### MAE
**Mean Absolute Error**

Measures the average absolute difference between actual and predicted prices.

### RMSE
**Root Mean Squared Error**

Measures prediction error while giving higher weight to larger errors.

### R² Score

Measures how well the model explains the variation in laptop prices.

The actual model performance metrics are displayed dynamically inside the Streamlit dashboard.

---

# 📌 Business Insights

The project converts data analysis into three important business areas:

## ⚠️ Risks

Examples of risks identified through market analysis:

- Strong competition between laptop brands
- Large price differences between specifications
- Changing customer preferences
- Different pricing patterns across processors and storage configurations

---

## 💡 Opportunities

Potential opportunities include:

- Growing demand for SSD-based laptops
- Student-focused laptop products
- Performance-based laptop pricing
- Premium laptop segments
- Region-specific product strategies

---

## 🎯 Executive Actions

Based on the analysis, businesses can:

- Optimize laptop pricing
- Identify high-value customer segments
- Improve product positioning
- Focus on high-demand specifications
- Develop targeted marketing strategies
- Use AI-assisted price estimation

---

# 🗂️ Dataset

The project uses a structured laptop market dataset containing information such as:

| Feature | Description |
|---|---|
| Laptop_ID | Unique laptop identifier |
| Brand | Laptop manufacturer |
| Model | Laptop model |
| Processor | Processor category |
| RAM_GB | RAM capacity |
| Storage_Type | SSD/HDD |
| Storage_GB | Storage capacity |
| GPU | Graphics processor |
| Screen_Size | Display size |
| Operating_System | Operating system |
| Weight_KG | Laptop weight |
| Battery_Hours | Estimated battery life |
| Warranty_Months | Warranty period |
| Customer_Segment | Target customer group |
| Region | Market region |
| Price | Laptop price |

> **Note:** The dataset is a simulated academic dataset created for demonstrating data analytics and machine learning techniques.

---

# 📊 Key Analytics

The dashboard provides analysis of:

- Brand-wise average price
- Brand market distribution
- Processor distribution
- Storage distribution
- Customer segment distribution
- RAM vs average price
- Storage vs average price
- Brand and processor pricing
- Laptop price prediction

---


---

# 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| 🐍 Python | Core programming language |
| 📊 Pandas | Data manipulation |
| 🔢 NumPy | Numerical operations |
| 📈 Matplotlib | Data visualization |
| 🎨 Seaborn | Statistical visualization |
| 🤖 Scikit-learn | Machine learning |
| 🖥️ Streamlit | Interactive dashboard |
| 🐙 GitHub | Version control and project hosting |

---

# 📁 Project Structure

```text
Laptop-Market-Analytics-AI/
│
├── Hajmina Patel_LaptopPricePredictor.py
│
├── requirements.txt
│
├── README.md
│
├── Project_Report.docx
│
└──Dataset.csv
```

---

# ⚙️ Installation

## Step 1 — Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

## Step 2 — Open the Project Folder

```bash
cd Laptop-Market-Analytics-AI
```

## Step 3 — Install Required Libraries

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Dashboard

Run the following command:

```bash
streamlit run Hajmina_LaptopPricePredictor.py
```

After running the command, Streamlit will open the dashboard in your web browser.

---

# 🔄 Project Workflow

```text
             LAPTOP MARKET DATA
                     │
                     ▼
              DATA CLEANING
                     │
                     ▼
             DATA ANALYTICS
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
   VISUAL ANALYTICS       BUSINESS INSIGHTS
          │                     │
          └──────────┬──────────┘
                     ▼
              MACHINE LEARNING
                     │
                     ▼
             PRICE PREDICTION
                     │
                     ▼
             STREAMLIT DASHBOARD
```

---

# 🌟 Main Highlights

### 📊 Data Analytics
Interactive analysis of laptop market trends.

### 🤖 Machine Learning
Random Forest based price prediction.

### 📈 Interactive Dashboard
Streamlit dashboard with filters and visualizations.

### 💰 Pricing Intelligence
Analysis of specifications affecting laptop prices.

### 🎯 Business Insights
Risks, opportunities and executive actions.

### 📥 Dataset Export
Users can download the processed dataset directly from the dashboard.

---

# 🔍 Why This Project Matters

Laptop pricing depends on multiple product characteristics.

Instead of looking at price as a single value, this project analyzes the relationship between:

```text
Brand
  +
Processor
  +
RAM
  +
Storage
  +
GPU
  +
Display
  +
Battery
  +
Customer Segment
  +
Region
       ↓
   LAPTOP PRICE
```

This approach demonstrates how data analytics and machine learning can support pricing and business decisions.

---

# 🎓 IBM Virtual Internship Project

**Program:** Data Analytics with AI Academic Internship 2026

**Project:** Laptop Market Analytics & AI Price Predictor

**Domain:** Data Analytics + Artificial Intelligence + Machine Learning

---

# 👨‍💻 Author

### Hajmina shabbir Patel

**Data Analytics & AI Project**

---

# 📜 Disclaimer

This project is developed for **academic and educational purposes**.

The dataset used in this project is simulated and is intended to demonstrate:

- Data analytics
- Machine learning
- Data visualization
- Business intelligence
- AI-based price prediction

The predicted prices should not be treated as real-world commercial pricing recommendations.

---

# ⭐ Project Summary

The **Laptop Market Analytics & AI Price Predictor** demonstrates a complete data analytics workflow starting from dataset generation and cleaning to visualization, machine learning, prediction, and business insights.

The project combines **Python, Pandas, NumPy, Scikit-learn, Seaborn, Matplotlib, and Streamlit** into a single interactive analytical application.

---

## 🚀 Future Enhancements

Future versions of the project can include:

- Real-world laptop datasets
- Live e-commerce price data
- Advanced machine learning models
- XGBoost price prediction
- Price trend forecasting
- Product recommendation system
- Automated model retraining
- Cloud deployment
- Real-time market monitoring

---

## ⭐ If You Like This Project

If this project helped you understand data analytics and machine learning, consider giving the repository a ⭐ on GitHub.

---

**Built with Python 🐍 | Data Analytics 📊 | Machine Learning 🤖 | Streamlit 🖥️**