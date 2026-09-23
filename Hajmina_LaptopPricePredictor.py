# ============================================================
# LAPTOP MARKET ANALYTICS & AI PRICE PREDICTOR
# Data Analytics with AI Academic Internship 2026
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 0. STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Laptop Market Analytics & AI Price Predictor",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Reproducibility
random.seed(42)
np.random.seed(42)

# Simple dashboard styling
st.markdown("""
<style>
    .main-title {
        font-size: 38px;
        font-weight: 700;
    }

    .sub-title {
        font-size: 18px;
        color: #666;
    }

    .insight-box {
        padding: 18px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# 1. PROJECT TITLE
# ============================================================

st.markdown(
    '<div class="main-title">💻 Laptop Market Analytics & AI Price Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">Data Analytics with AI Academic Internship 2026</div>',
    unsafe_allow_html=True
)

st.markdown(
    "**Partners:** IBM SkillsBuild, BharatCares, CSR Box & AICTE"
)

st.divider()


# ============================================================
# 2. SYNTHETIC COMMERCIAL DATA GENERATION
# ============================================================

@st.cache_data
def generate_dataset():

    total_rows = 500

    brands = [
        "Dell",
        "HP",
        "Lenovo",
        "ASUS",
        "Acer",
        "Apple",
        "MSI",
        "Samsung"
    ]

    processors = [
        "Intel i3",
        "Intel i5",
        "Intel i7",
        "AMD Ryzen 3",
        "AMD Ryzen 5",
        "AMD Ryzen 7",
        "Apple M-Series"
    ]

    models = {
        "Dell": ["Inspiron 15", "Vostro 15", "Latitude 14"],
        "HP": ["Pavilion 15", "HP 15", "Victus 15"],
        "Lenovo": ["IdeaPad Slim 3", "ThinkBook 14", "IdeaPad Gaming 3"],
        "ASUS": ["VivoBook 15", "ZenBook 14", "TUF Gaming F15"],
        "Acer": ["Aspire 5", "Aspire 7", "Swift 3"],
        "Apple": ["MacBook Air", "MacBook Pro"],
        "MSI": ["Modern 14", "Katana 15", "Thin GF63"],
        "Samsung": ["Galaxy Book", "Galaxy Book3", "Galaxy Book Pro"]
    }

    # Exact brand distribution:
    # Dell = 160 / 500 = 32%
    brand_counts = {
        "Dell": 160,
        "HP": 50,
        "Lenovo": 50,
        "ASUS": 50,
        "Acer": 50,
        "Apple": 40,
        "MSI": 50,
        "Samsung": 50
    }

    brand_base_price = {
        "Dell": 57500,
        "HP": 52000,
        "Lenovo": 55000,
        "ASUS": 54000,
        "Acer": 45000,
        "Apple": 115000,
        "MSI": 62000,
        "Samsung": 49000
    }

    processor_addition = {
        "Intel i3": -8000,
        "Intel i5": 0,
        "Intel i7": 15000,
        "AMD Ryzen 3": -7000,
        "AMD Ryzen 5": 4000,
        "AMD Ryzen 7": 13000,
        "Apple M-Series": 0
    }

    ram_options = [8, 16, 32, 64]
    ram_addition = {
        8: 0,
        16: 7000,
        32: 18000,
        64: 32000
    }

    storage_sizes = [256, 512, 1024, 2048]
    storage_addition = {
        256: 0,
        512: 5000,
        1024: 10000,
        2048: 18000
    }

    gpu_options = [
        "Integrated",
        "NVIDIA GTX",
        "NVIDIA RTX",
        "AMD Radeon"
    ]

    gpu_addition = {
        "Integrated": 0,
        "NVIDIA GTX": 7000,
        "NVIDIA RTX": 18000,
        "AMD Radeon": 6000
    }

    screen_sizes = [13.3, 14.0, 15.6, 16.0, 17.3]

    operating_systems = [
        "Windows 11",
        "macOS",
        "Linux"
    ]

    customer_segments = [
        "Student",
        "Professional",
        "Business",
        "Gaming"
    ]

    regions = [
        "West",
        "North",
        "South",
        "East"
    ]

    # Exact 75% SSD distribution
    storage_types = ["SSD"] * 375 + ["HDD"] * 125
    random.shuffle(storage_types)

    data = []
    storage_index = 0

    for brand, count in brand_counts.items():

        for _ in range(count):

            if brand == "Apple":
                processor = "Apple M-Series"
                operating_system = "macOS"
                gpu = "Integrated"
            else:
                processor = random.choices(
                    processors[:-1],
                    weights=[12, 28, 18, 12, 20, 10]
                )[0]

                operating_system = random.choices(
                    ["Windows 11", "Linux"],
                    weights=[90, 10]
                )[0]

                gpu = random.choices(
                    gpu_options,
                    weights=[55, 15, 15, 15]
                )[0]

            ram = random.choices(
                ram_options,
                weights=[45, 40, 12, 3]
            )[0]

            storage_gb = random.choices(
                storage_sizes,
                weights=[20, 50, 25, 5]
            )[0]

            storage_type = storage_types[storage_index]
            storage_index += 1

            model = random.choice(models[brand])

            screen_size = random.choice(screen_sizes)

            if screen_size <= 14:
                weight = round(random.uniform(1.1, 1.6), 2)
            elif screen_size <= 16:
                weight = round(random.uniform(1.5, 2.2), 2)
            else:
                weight = round(random.uniform(2.0, 2.8), 2)

            battery_hours = round(
                random.uniform(5.5, 12.0), 1
            )

            warranty = random.choice([6, 12, 24])

            customer_segment = random.choices(
                customer_segments,
                weights=[35, 30, 20, 15]
            )[0]

            region = random.choice(regions)

            # ------------------------------------------------
            # Synthetic commercial pricing formula
            # ------------------------------------------------

            price = (
                brand_base_price[brand]
                + processor_addition[processor]
                + ram_addition[ram]
                + storage_addition[storage_gb]
                + gpu_addition[gpu]
            )

            # SSD premium
            if storage_type == "SSD":
                price += 5000

            # Small screen premium adjustment
            if screen_size <= 14:
                price += 3000

            # Random commercial market variation
            price += np.random.normal(0, 4500)

            # Keep price realistic
            price = max(28000, price)

            data.append({
                "Laptop_ID": f"LAP{len(data)+1:04d}",
                "Brand": brand,
                "Model": model,
                "Processor": processor,
                "RAM_GB": ram,
                "Storage_Type": storage_type,
                "Storage_GB": storage_gb,
                "GPU": gpu,
                "Screen_Size": screen_size,
                "Operating_System": operating_system,
                "Weight_KG": weight,
                "Battery_Hours": battery_hours,
                "Warranty_Months": warranty,
                "Customer_Segment": customer_segment,
                "Region": region,
                "Price": round(price, 2)
            })

    df = pd.DataFrame(data)

    # --------------------------------------------------------
    # Benchmark calibration required by project specification
    # --------------------------------------------------------

    # Apple benchmark ≈ ₹115,000
    apple_mask = df["Brand"] == "Apple"

    apple_mean = df.loc[apple_mask, "Price"].mean()

    if apple_mean > 0:
        df.loc[apple_mask, "Price"] = (
            df.loc[apple_mask, "Price"]
            * (115000 / apple_mean)
        )

    # Overall benchmark = ₹62,450
    current_mean = df["Price"].mean()

    difference = 62450 - current_mean

    non_apple_mask = df["Brand"] != "Apple"

    df.loc[non_apple_mask, "Price"] += difference

    # Safety floor
    df["Price"] = df["Price"].clip(lower=25000)

    df["Price"] = df["Price"].round(2)

    return df


df = generate_dataset()


# ============================================================
# 3. DATA VALIDATION / CLEANING
# ============================================================

def clean_data(dataframe):

    clean_df = dataframe.copy()

    # Remove duplicate records
    clean_df = clean_df.drop_duplicates()

    # Remove impossible prices
    clean_df = clean_df[
        (clean_df["Price"] > 0)
    ]

    # Handle missing numerical values
    numerical_columns = [
        "RAM_GB",
        "Storage_GB",
        "Screen_Size",
        "Weight_KG",
        "Battery_Hours",
        "Warranty_Months",
        "Price"
    ]

    for column in numerical_columns:
        clean_df[column] = clean_df[column].fillna(
            clean_df[column].median()
        )

    # Handle missing categorical values
    categorical_columns = [
        "Brand",
        "Model",
        "Processor",
        "Storage_Type",
        "GPU",
        "Operating_System",
        "Customer_Segment",
        "Region"
    ]

    for column in categorical_columns:
        clean_df[column] = clean_df[column].fillna("Unknown")

    return clean_df


df = clean_data(df)


# ============================================================
# 4. GLOBAL KPIs
# ============================================================

average_price = df["Price"].mean()
total_brands = df["Brand"].nunique()

brand_share = (
    df["Brand"]
    .value_counts(normalize=True)
    .mul(100)
)

market_leader = brand_share.idxmax()
market_leader_share = brand_share.max()

ssd_share = (
    df["Storage_Type"]
    .value_counts(normalize=True)
    .get("SSD", 0)
    * 100
)


# ============================================================
# 5. SIDEBAR FILTERS
# ============================================================

st.sidebar.header("📊 Interactive Marketplace Filters")

st.sidebar.write(
    "Filter the laptop market dataset using the controls below."
)

selected_brands = st.sidebar.multiselect(
    "Select Manufacturer Brand",
    options=sorted(df["Brand"].unique()),
    default=sorted(df["Brand"].unique())
)

selected_processors = st.sidebar.multiselect(
    "Select Processor Tier",
    options=sorted(df["Processor"].unique()),
    default=sorted(df["Processor"].unique())
)

selected_segments = st.sidebar.multiselect(
    "Select Customer Segment",
    options=sorted(df["Customer_Segment"].unique()),
    default=sorted(df["Customer_Segment"].unique())
)

selected_regions = st.sidebar.multiselect(
    "Select Region",
    options=sorted(df["Region"].unique()),
    default=sorted(df["Region"].unique())
)


filtered_df = df[
    df["Brand"].isin(selected_brands)
    & df["Processor"].isin(selected_processors)
    & df["Customer_Segment"].isin(selected_segments)
    & df["Region"].isin(selected_regions)
]


# ============================================================
# 6. LEVEL 1 — EXECUTIVE KPI DASHBOARD
# ============================================================

st.subheader("📊 Performance Trackers & KPIs")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.metric(
        "Average Market Price",
        f"₹{average_price:,.2f}"
    )

with kpi2:
    st.metric(
        "Total Brands",
        total_brands
    )

with kpi3:
    st.metric(
        "Market Leader",
        market_leader
    )

with kpi4:
    st.metric(
        "SSD Market Share",
        f"{ssd_share:.1f}%"
    )


st.divider()


# ============================================================
# 7. TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Market Analytics",
    "📈 Pricing Trends",
    "🤖 AI Price Predictor",
    "🧠 Business Insights",
    "📋 Dataset"
])


# ============================================================
# TAB 1 — MARKET ANALYTICS
# ============================================================

with tab1:

    st.subheader("📊 Laptop Market Analytics")

    if filtered_df.empty:

        st.warning(
            "No records match the selected filters."
        )

    else:

        col1, col2 = st.columns(2)

        # ----------------------------------------------------
        # Brand-wise Average Price
        # ----------------------------------------------------

        with col1:

            st.markdown("### Brand-wise Average Price")

            brand_prices = (
                filtered_df
                .groupby("Brand")["Price"]
                .mean()
                .sort_values(ascending=False)
            )

            fig, ax = plt.subplots(figsize=(8, 5))

            sns.barplot(
                x=brand_prices.values,
                y=brand_prices.index,
                ax=ax
            )

            ax.set_xlabel("Average Price (₹)")
            ax.set_ylabel("Brand")
            ax.set_title("Average Laptop Price by Brand")

            st.pyplot(fig)
            plt.close(fig)

        # ----------------------------------------------------
        # Processor Distribution
        # ----------------------------------------------------

        with col2:

            st.markdown("### Processor Market Distribution")

            processor_counts = (
                filtered_df["Processor"]
                .value_counts()
            )

            fig, ax = plt.subplots(figsize=(8, 5))

            processor_counts.plot(
                kind="bar",
                ax=ax
            )

            ax.set_xlabel("Processor")
            ax.set_ylabel("Number of Laptops")
            ax.set_title("Processor Tier Distribution")

            plt.xticks(rotation=45)

            st.pyplot(fig)
            plt.close(fig)

        # ----------------------------------------------------
        # Storage Split
        # ----------------------------------------------------

        col3, col4 = st.columns(2)

        with col3:

            st.markdown("### Storage Architecture")

            storage_counts = (
                filtered_df["Storage_Type"]
                .value_counts()
            )

            fig, ax = plt.subplots(figsize=(6, 5))

            ax.pie(
                storage_counts.values,
                labels=storage_counts.index,
                autopct="%1.1f%%",
                startangle=90
            )

            ax.set_title("SSD vs HDD Market Split")

            st.pyplot(fig)
            plt.close(fig)

        # ----------------------------------------------------
        # Customer Segment
        # ----------------------------------------------------

        with col4:

            st.markdown("### Customer Segment Distribution")

            segment_counts = (
                filtered_df["Customer_Segment"]
                .value_counts()
            )

            fig, ax = plt.subplots(figsize=(6, 5))

            segment_counts.plot(
                kind="bar",
                ax=ax
            )

            ax.set_xlabel("Customer Segment")
            ax.set_ylabel("Laptop Count")
            ax.set_title("Customer Segment Distribution")

            plt.xticks(rotation=30)

            st.pyplot(fig)
            plt.close(fig)


# ============================================================
# TAB 2 — PRICING TRENDS
# ============================================================

with tab2:

    st.subheader("📈 Pricing & Hardware Trend Analysis")

    if filtered_df.empty:

        st.warning(
            "No records available for the selected filters."
        )

    else:

        col1, col2 = st.columns(2)

        # ----------------------------------------------------
        # RAM vs Price
        # ----------------------------------------------------

        with col1:

            st.markdown("### RAM vs Average Price")

            ram_price = (
                filtered_df
                .groupby("RAM_GB")["Price"]
                .mean()
                .reset_index()
            )

            fig, ax = plt.subplots(figsize=(7, 5))

            sns.lineplot(
                data=ram_price,
                x="RAM_GB",
                y="Price",
                marker="o",
                ax=ax
            )

            ax.set_xlabel("RAM (GB)")
            ax.set_ylabel("Average Price (₹)")
            ax.set_title("RAM Capacity vs Average Price")

            st.pyplot(fig)
            plt.close(fig)

        # ----------------------------------------------------
        # Storage vs Price
        # ----------------------------------------------------

        with col2:

            st.markdown("### Storage Type vs Price")

            storage_price = (
                filtered_df
                .groupby("Storage_Type")["Price"]
                .mean()
                .reset_index()
            )

            fig, ax = plt.subplots(figsize=(7, 5))

            sns.barplot(
                data=storage_price,
                x="Storage_Type",
                y="Price",
                ax=ax
            )

            ax.set_xlabel("Storage Type")
            ax.set_ylabel("Average Price (₹)")
            ax.set_title("Storage Type vs Average Price")

            st.pyplot(fig)
            plt.close(fig)

        # ----------------------------------------------------
        # Brand × Processor Heatmap
        # ----------------------------------------------------

        st.markdown("### 🔥 Brand × Processor Price Heatmap")

        heatmap_data = pd.pivot_table(
            filtered_df,
            values="Price",
            index="Brand",
            columns="Processor",
            aggfunc="mean"
        )

        fig, ax = plt.subplots(
            figsize=(12, 6)
        )

        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt=".0f",
            cmap="Blues",
            ax=ax
        )

        ax.set_title(
            "Average Laptop Price by Brand and Processor"
        )

        st.pyplot(fig)
        plt.close(fig)


# ============================================================
# 8. MACHINE LEARNING MODEL
# ============================================================

@st.cache_resource
def train_price_model(dataframe):

    model_df = dataframe.copy()

    target = "Price"

    features = [
        "Brand",
        "Processor",
        "RAM_GB",
        "Storage_Type",
        "Storage_GB",
        "GPU",
        "Screen_Size",
        "Operating_System",
        "Weight_KG",
        "Battery_Hours",
        "Warranty_Months",
        "Customer_Segment",
        "Region"
    ]

    X = model_df[features]
    y = model_df[target]

    categorical_features = [
        "Brand",
        "Processor",
        "Storage_Type",
        "GPU",
        "Operating_System",
        "Customer_Segment",
        "Region"
    ]

    numerical_features = [
        "RAM_GB",
        "Storage_GB",
        "Screen_Size",
        "Weight_KG",
        "Battery_Hours",
        "Warranty_Months"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_features
            ),
            (
                "numerical",
                "passthrough",
                numerical_features
            )
        ]
    )

    model = RandomForestRegressor(
        n_estimators=250,
        random_state=42,
        max_depth=12,
        min_samples_split=3
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    return pipeline, mae, rmse, r2


model, mae, rmse, r2 = train_price_model(df)


# ============================================================
# TAB 3 — AI PRICE PREDICTOR
# ============================================================

with tab3:

    st.subheader("🤖 AI Laptop Price Predictor")

    st.write(
        "Configure a laptop specification and the trained "
        "machine-learning model will estimate its market price."
    )

    pred_col1, pred_col2 = st.columns(2)

    with pred_col1:

        prediction_brand = st.selectbox(
            "Manufacturer Brand",
            sorted(df["Brand"].unique())
        )

        prediction_processor = st.selectbox(
            "Processor",
            sorted(df["Processor"].unique())
        )

        prediction_ram = st.selectbox(
            "RAM (GB)",
            sorted(df["RAM_GB"].unique())
        )

        prediction_storage_type = st.selectbox(
            "Storage Type",
            sorted(df["Storage_Type"].unique())
        )

        prediction_storage = st.selectbox(
            "Storage Capacity (GB)",
            sorted(df["Storage_GB"].unique())
        )

        prediction_gpu = st.selectbox(
            "GPU",
            sorted(df["GPU"].unique())
        )

    with pred_col2:

        prediction_screen = st.selectbox(
            "Screen Size",
            sorted(df["Screen_Size"].unique())
        )

        prediction_os = st.selectbox(
            "Operating System",
            sorted(df["Operating_System"].unique())
        )

        prediction_weight = st.slider(
            "Weight (KG)",
            min_value=1.0,
            max_value=3.0,
            value=1.8,
            step=0.1
        )

        prediction_battery = st.slider(
            "Battery Hours",
            min_value=5.0,
            max_value=12.0,
            value=8.0,
            step=0.5
        )

        prediction_warranty = st.selectbox(
            "Warranty (Months)",
            sorted(df["Warranty_Months"].unique())
        )

        prediction_segment = st.selectbox(
            "Customer Segment",
            sorted(df["Customer_Segment"].unique())
        )

        prediction_region = st.selectbox(
            "Region",
            sorted(df["Region"].unique())
        )

    predict_button = st.button(
        "🔮 Predict Laptop Price",
        type="primary"
    )

    if predict_button:

        input_data = pd.DataFrame({
            "Brand": [prediction_brand],
            "Processor": [prediction_processor],
            "RAM_GB": [prediction_ram],
            "Storage_Type": [prediction_storage_type],
            "Storage_GB": [prediction_storage],
            "GPU": [prediction_gpu],
            "Screen_Size": [prediction_screen],
            "Operating_System": [prediction_os],
            "Weight_KG": [prediction_weight],
            "Battery_Hours": [prediction_battery],
            "Warranty_Months": [prediction_warranty],
            "Customer_Segment": [prediction_segment],
            "Region": [prediction_region]
        })

        predicted_price = model.predict(
            input_data
        )[0]

        st.success(
            f"🎯 AI Predicted Market Price: "
            f"₹{predicted_price:,.2f}"
        )

        st.caption(
            "Prediction generated using a Random Forest "
            "Regression model trained on the project dataset."
        )

    st.divider()

    st.markdown("### 📊 Model Performance")

    model_col1, model_col2, model_col3 = st.columns(3)

    with model_col1:
        st.metric(
            "MAE",
            f"₹{mae:,.0f}"
        )

    with model_col2:
        st.metric(
            "RMSE",
            f"₹{rmse:,.0f}"
        )

    with model_col3:
        st.metric(
            "R² Score",
            f"{r2:.3f}"
        )

    st.info(
        "MAE shows average prediction error, RMSE penalizes "
        "larger errors, and R² indicates how much variation "
        "in laptop price is explained by the model."
    )


# ============================================================
# TAB 4 — BUSINESS INSIGHTS
# ============================================================

with tab4:

    st.subheader(
        "🧠 AI Business Insights Framework"
    )

    # --------------------------------------------------------
    # Calculate data-driven insights
    # --------------------------------------------------------

    intel_i5_price = df.loc[
        df["Processor"] == "Intel i5",
        "Price"
    ].mean()

    amd_price = df[
        df["Processor"].str.contains("AMD")
    ]["Price"].mean()

    overall_price = df["Price"].mean()

    student_amd = df[
        (df["Customer_Segment"] == "Student")
        & (df["Processor"].str.contains("AMD"))
    ].shape[0]

    student_total = df[
        df["Customer_Segment"] == "Student"
    ].shape[0]

    student_amd_share = (
        student_amd / student_total * 100
        if student_total > 0
        else 0
    )

    # --------------------------------------------------------
    # RISK
    # --------------------------------------------------------

    st.error(
        f"""
        🚨 **RISK — Processor Price Pressure**

        Intel i5 laptops have an average market price of
        **₹{intel_i5_price:,.0f}**.

        The business should continuously monitor procurement
        costs and availability for the mid-range Intel segment.
        """
    )

    # --------------------------------------------------------
    # OPPORTUNITY
    # --------------------------------------------------------

    st.success(
        f"""
        🎯 **OPPORTUNITY — Student AMD Segment**

        AMD-based laptops represent approximately
        **{student_amd_share:.1f}%** of the student-segment
        records in the simulated market dataset.

        This indicates a potential opportunity for affordable
        AMD-based student laptop packages.
        """
    )

    # --------------------------------------------------------
    # EXECUTIVE ACTION
    # --------------------------------------------------------

    st.info(
        """
        📋 **EXECUTIVE MANAGEMENT ACTION PLAN**

        **1. Procurement Strategy:**  
        Monitor Intel i5 supply and pricing before major
        purchasing cycles.

        **2. AMD Portfolio Expansion:**  
        Evaluate additional AMD Ryzen 5 and Ryzen 7 inventory
        for mid-range customers.

        **3. Student Packages:**  
        Develop student-oriented packages combining
        8/16GB RAM, SSD storage and warranty coverage.

        **4. Pricing Strategy:**  
        Use the AI price predictor as a decision-support
        tool when evaluating new laptop configurations.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # Premium ecosystem insight
    # --------------------------------------------------------

    apple_average = df.loc[
        df["Brand"] == "Apple",
        "Price"
    ].mean()

    st.markdown("### 🍎 Premium Ecosystem Insight")

    st.write(
        f"Apple laptops show an average simulated market price "
        f"of **₹{apple_average:,.2f}**, reflecting their premium "
        f"position within the dataset."
    )


# ============================================================
# TAB 5 — DATASET
# ============================================================

with tab5:

    st.subheader("📋 Project Dataset")

    st.write(
        f"Total cleaned records: **{len(df)}**"
    )

    st.dataframe(
        df,
        use_container_width=True,
        height=500
    )

    # CSV download
    csv_data = df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="⬇️ Download Dataset CSV",
        data=csv_data,
        file_name="laptop_market_dataset.csv",
        mime="text/csv"
    )


# ============================================================
# 9. FOOTER
# ============================================================

st.divider()

st.caption(
    "Laptop Market Analytics & AI Price Predictor | "
    "Data Analytics with AI Academic Internship 2026"
)