
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime

st.set_page_config(page_title="Data Wrangler & Visualizer", layout="wide")

st.title("AI-Assisted Data Wrangler & Visualizer")

st.write(
"This application allows users to upload datasets, clean and transform data, "
"visualize patterns, and receive AI-assisted insights."
)
# -----------------------------
# SESSION STORAGE
# -----------------------------

if "df" not in st.session_state:
    st.session_state.df = None

if "log" not in st.session_state:
    st.session_state.log = []

# -----------------------------
# SIDEBAR
# -----------------------------

menu = st.sidebar.radio(
    "Navigation",
    [
        "Upload & Overview",
        "Cleaning",
        "Transformation",
        "Visualization",
        "AI Insights",
        "Export"
    ]
)

# -----------------------------
# PAGE 1 – UPLOAD
# -----------------------------

if menu == "Upload & Overview":

    st.header("Upload Dataset")

    file = st.file_uploader("Upload CSV, Excel, or JSON", type=["csv","xlsx","json"])

    if file is not None:

        if file.name.endswith(".csv"):
            df = pd.read_csv(file)

        elif file.name.endswith(".xlsx"):
            df = pd.read_excel(file)

        else:
            df = pd.read_json(file)

        st.session_state.df = df
        st.session_state.log.append("Dataset uploaded")

    if st.session_state.df is not None:

        df = st.session_state.df

        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        col1, col2, col3 = st.columns(3)

        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])
        col3.metric("Missing Values", df.isnull().sum().sum())

        st.subheader("Column Types")
        st.write(df.dtypes)

        st.subheader("Missing Values per Column")
        st.write(df.isnull().sum())

        st.subheader("Summary Statistics")
        st.write(df.describe())

# -----------------------------
# PAGE 2 – CLEANING
# -----------------------------

elif menu == "Cleaning":

    df = st.session_state.df

    if df is None:
        st.warning("Upload dataset first")

    else:

        st.header("Data Cleaning")

        st.subheader("Handle Missing Values")

        method = st.selectbox(
            "Method",
            [
                "Drop Rows",
                "Fill Mean",
                "Fill Median",
                "Fill Mode"
            ]
        )

        if st.button("Apply"):

            if method == "Drop Rows":
                df = df.dropna()

            elif method == "Fill Mean":
                df = df.fillna(df.mean(numeric_only=True))

            elif method == "Fill Median":
                df = df.fillna(df.median(numeric_only=True))

            elif method == "Fill Mode":
                df = df.fillna(df.mode().iloc[0])

            st.session_state.df = df
            st.session_state.log.append(f"Missing values handled: {method}")

            st.success("Missing values processed")

        st.subheader("Duplicates")

        duplicates = df.duplicated().sum()

        st.write("Duplicate rows:", duplicates)

        if st.button("Remove duplicates"):

            df = df.drop_duplicates()

            st.session_state.df = df
            st.session_state.log.append("Duplicates removed")

            st.success("Duplicates removed")

# -----------------------------
# PAGE 3 – TRANSFORMATION
# -----------------------------

elif menu == "Transformation":

    df = st.session_state.df

    if df is None:
        st.warning("Upload dataset first")

    else:

        st.header("Data Transformation")

        st.subheader("Rename Column")

        column = st.selectbox("Select column", df.columns)

        new_name = st.text_input("New column name")

        if st.button("Rename column"):

            df = df.rename(columns={column:new_name})

            st.session_state.df = df
            st.session_state.log.append(f"Column renamed {column} -> {new_name}")

            st.success("Column renamed")

        st.subheader("Convert Data Type")

        column2 = st.selectbox("Column", df.columns)

        dtype = st.selectbox("Convert to", ["int","float","string"])

        if st.button("Convert type"):

            try:
                df[column2] = df[column2].astype(dtype)

                st.session_state.df = df
                st.session_state.log.append(f"{column2} converted to {dtype}")

                st.success("Conversion successful")

            except:
                st.error("Conversion failed")

        st.subheader("Outlier Detection (IQR)")

        numeric_cols = df.select_dtypes(include=np.number).columns

        col = st.selectbox("Numeric column", numeric_cols)

        if st.button("Remove outliers"):

            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)

            IQR = Q3 - Q1

            df = df[(df[col] >= Q1 - 1.5*IQR) &
                    (df[col] <= Q3 + 1.5*IQR)]

            st.session_state.df = df
            st.session_state.log.append(f"Outliers removed from {col}")

            st.success("Outliers removed")

        st.subheader("Normalization")

        column3 = st.selectbox("Column for scaling", numeric_cols)

        method = st.selectbox(
            "Scaling method",
            [
                "Min-Max",
                "Z-score"
            ]
        )

        if st.button("Apply scaling"):

            if method == "Min-Max":

                df[column3] = (
                    df[column3] - df[column3].min()
                ) / (
                    df[column3].max() - df[column3].min()
                )

            else:

                df[column3] = (
                    df[column3] - df[column3].mean()
                ) / df[column3].std()

            st.session_state.df = df
            st.session_state.log.append(f"{column3} scaled using {method}")

            st.success("Scaling applied")

# -----------------------------
# PAGE 4 – VISUALIZATION
# -----------------------------

elif menu == "Visualization":

    df = st.session_state.df

    if df is None:
        st.warning("Upload dataset first")

    else:
        st.header("Data Visualization")

    
        sample_df = df.sample(min(100, len(df)))

        chart = st.selectbox(
            "Chart type",
            [
                "Histogram",
                "Scatter",
                "Line",
                "Bar",
                "Box Plot",
                "Correlation Heatmap"
            ]
        )

        fig, ax = plt.subplots(figsize=(6,4))  

        if chart == "Histogram":

            col = st.selectbox("Column", df.select_dtypes(include='number').columns)

            ax.hist(sample_df[col].dropna(), bins=20)
            ax.set_title(f"Distribution of {col}")

        elif chart == "Scatter":

            numeric_cols = df.select_dtypes(include='number').columns

            x = st.selectbox("X axis", numeric_cols)
            y = st.selectbox("Y axis", numeric_cols)

            ax.scatter(sample_df[x], sample_df[y])
            ax.set_title(f"{y} vs {x}")

        elif chart == "Line":

            numeric_cols = df.select_dtypes(include='number').columns

            x = st.selectbox("X axis", numeric_cols)
            y = st.selectbox("Y axis", numeric_cols)

            
            sorted_df = sample_df.sort_values(by=x)

            ax.plot(sorted_df[x], sorted_df[y])
            ax.set_title(f"{y} over {x}")

        elif chart == "Bar":

            cat_cols = df.select_dtypes(include='object').columns
            num_cols = df.select_dtypes(include='number').columns

            x = st.selectbox("Category column", cat_cols)
            y = st.selectbox("Value column", num_cols)

            grouped = df.groupby(x)[y].mean()

            ax.bar(grouped.index, grouped.values)
            ax.set_title(f"Average {y} by {x}")

            plt.xticks(rotation=45)

        elif chart == "Box Plot":

            col = st.selectbox("Column", df.select_dtypes(include='number').columns)

            ax.boxplot(sample_df[col].dropna())
            ax.set_title(f"Boxplot of {col}")

        elif chart == "Correlation Heatmap":

            corr = df.corr(numeric_only=True)

            cax = ax.matshow(corr)
            fig.colorbar(cax)

            ax.set_xticks(range(len(corr.columns)))
            ax.set_yticks(range(len(corr.columns)))

            ax.set_xticklabels(corr.columns, rotation=90)
            ax.set_yticklabels(corr.columns)

            ax.set_title("Correlation Heatmap")

        st.pyplot(fig)
          

           
          
        
        # -----------------------------
# PAGE 5 – AI INSIGHTS
# -----------------------------

elif menu == "AI Insights":

    df = st.session_state.df

    if df is None:
        st.warning("Upload dataset first")

    else:

        st.header("AI Data Insights")

        st.write("AI analyzing your dataset...")

        numeric = df.select_dtypes(include=np.number)

        if numeric.empty:
            st.warning("No numeric columns found for AI analysis")

        else:

            st.subheader("AI Dataset Summary")

            st.write("Rows:", df.shape[0])
            st.write("Columns:", df.shape[1])

            st.subheader("AI Observations")

            for col in numeric.columns:

                mean = numeric[col].mean()
                std = numeric[col].std()

                st.write(
                    f"Column **{col}** has average value {round(mean,2)} and standard deviation {round(std,2)}"
                )

            st.subheader("AI Correlation Detection")

            corr = numeric.corr()

            found = False

            for c1 in corr.columns:
                for c2 in corr.columns:

                    if c1 != c2 and abs(corr.loc[c1,c2]) > 0.7:

                        st.write(
                            f"Strong correlation detected between **{c1}** and **{c2}**"
                        )

                        found = True

            if not found:
                st.write("No strong correlations detected")

# -----------------------------
# PAGE 6 – EXPORT
# -----------------------------

elif menu == "Export":

    df = st.session_state.df

    if df is None:
        st.warning("Upload dataset first")

    else:

        st.header("Export Data")

        csv = df.to_csv(index=False)

        st.download_button(
            "Download Clean Dataset",
            csv,
            "clean_dataset.csv",
            "text/csv"
        )

        report = {
            "transformations": st.session_state.log,
            "time": str(datetime.now())
        }

        report_json = json.dumps(report, indent=4)

        st.download_button(
            "Download Transformation Report",
            report_json,
            "transformation_report.json",
            "application/json"
        )

        st.subheader("Transformation Log")

        for step in st.session_state.log:
            st.write("-", step)


  
  



