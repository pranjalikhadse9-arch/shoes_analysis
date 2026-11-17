
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

from scipy.stats import t, norm

st.set_page_config(
    page_title="Shoes Data Analysis",
    page_icon="👟",
    layout="wide"
)

st.title("👟 SHOES Dataset - Statistical Analysis")
st.write("Data Analysis using Python, Pandas, NumPy, Seaborn and Streamlit")

# ---------------- DATASET ----------------

st.sidebar.header("Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload SHOES.csv",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    try:
        df = pd.read_csv("SHOES.csv")
    except FileNotFoundError:
        st.info("Please upload your SHOES.csv dataset.")
        st.stop()

# ---------------- DATA CLEANING ----------------

for col in ["Sales", "Inventory", "Returns"]:
    if col in df.columns:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(r"[$,]", "", regex=True),
            errors="coerce"
        )

st.sidebar.success("Dataset loaded successfully!")

# ---------------- SIDEBAR MENU ----------------

menu = st.sidebar.selectbox(
    "Select Analysis",
    [
        "Dataset Overview",
        "Statistical Measures",
        "KDE Plot",
        "Outlier Detection",
        "Standard Deviation & Variance",
        "Correlation",
        "Percentiles",
        "Histogram",
        "Scatter Plot",
        "Pairplot",
        "Swarmplot",
        "Confidence Interval",
        "Hypothesis Testing"
    ]
)

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

# ---------------- DATASET OVERVIEW ----------------

if menu == "Dataset Overview":

    st.header("Dataset Overview")

    c1, c2, c3 = st.columns(3)

    c1.metric("Total Rows", df.shape[0])
    c2.metric("Total Columns", df.shape[1])
    c3.metric("Missing Values", df.isnull().sum().sum())

    st.subheader("Dataset")
    st.dataframe(df, use_container_width=True)

    st.subheader("Missing Values")
    st.dataframe(df.isnull().sum())

    st.subheader("Data Types")
    st.dataframe(df.dtypes.astype(str))

    st.subheader("Unique Sales Values")

    if "Sales" in df.columns:
        st.write(df.Sales.unique())

    st.subheader("Descriptive Statistics")
    st.dataframe(df.describe(), use_container_width=True)


# ---------------- STATISTICAL MEASURES ----------------

elif menu == "Statistical Measures":

    st.header("Mean, Median and Mode")

    selected = st.multiselect(
        "Select Columns",
        numeric_cols,
        default=numeric_cols
    )

    if selected:
        result = pd.DataFrame({
            "Mean": df[selected].mean(),
            "Median": df[selected].median(),
            "Mode": [
                df[col].mode().tolist() for col in selected
            ]
        })

        st.dataframe(result, use_container_width=True)


# ---------------- KDE PLOT ----------------

elif menu == "KDE Plot":

    st.header("KDE Distribution Plot")

    selected = st.multiselect(
        "Select Columns",
        numeric_cols,
        default=numeric_cols
    )

    if selected:
        fig, ax = plt.subplots(figsize=(10, 5))

        for col in selected:
            sb.kdeplot(data=df, x=col, ax=ax, label=col)

        ax.legend()
        ax.set_title("KDE Plot")
        st.pyplot(fig)
        plt.close(fig)


# ---------------- OUTLIER DETECTION ----------------

elif menu == "Outlier Detection":

    st.header("Outlier Detection using IQR")

    selected = st.multiselect(
        "Select Columns for Outlier Removal",
        numeric_cols,
        default=numeric_cols
    )

    st.subheader("Before Outlier Removal")

    for col in selected:
        fig, ax = plt.subplots()
        sb.boxplot(x=df[col], ax=ax)
        ax.set_title(f"{col} - Before Outlier Removal")
        st.pyplot(fig)
        plt.close(fig)

    if st.button("Remove Outliers"):

        original_rows = len(df)
        clean_df = df.copy()

        for col in selected:

            q1 = clean_df[col].quantile(0.25)
            q3 = clean_df[col].quantile(0.75)

            iqr = q3 - q1

            c1 = q1 - 1.5 * iqr
            c2 = q3 + 1.5 * iqr

            clean_df = clean_df[
                (clean_df[col] >= c1) &
                (clean_df[col] <= c2)
            ]

        removed = original_rows - len(clean_df)

        st.success("Outliers removed successfully!")

        st.metric("Original Rows", original_rows)
        st.metric("Rows After Removal", len(clean_df))
        st.metric("Removed Rows", removed)

        st.subheader("After Outlier Removal")

        for col in selected:
            fig, ax = plt.subplots()
            sb.boxplot(x=clean_df[col], ax=ax)
            ax.set_title(f"{col} - After Outlier Removal")
            st.pyplot(fig)
            plt.close(fig)

        st.subheader("Cleaned Dataset")
        st.dataframe(clean_df, use_container_width=True)

        st.download_button(
            "Download Cleaned Dataset",
            clean_df.to_csv(index=False),
            file_name="SHOES_Cleaned.csv",
            mime="text/csv"
        )


# ---------------- STANDARD DEVIATION & VARIANCE ----------------

elif menu == "Standard Deviation & Variance":

    st.header("Standard Deviation and Variance")

    selected = st.multiselect(
        "Select Columns",
        numeric_cols,
        default=numeric_cols
    )

    if selected:

        result = pd.DataFrame({
            "Standard Deviation": df[selected].std(),
            "Variance": df[selected].var()
        })

        st.dataframe(result, use_container_width=True)


# ---------------- CORRELATION ----------------

elif menu == "Correlation":

    st.header("Correlation Matrix")

    corr = df.corr(numeric_only=True)

    st.subheader("Correlation Values")
    st.dataframe(corr)

    fig, ax = plt.subplots(figsize=(10, 6))

    sb.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        ax=ax
    )

    st.pyplot(fig)
    plt.close(fig)


# ---------------- PERCENTILES ----------------

elif menu == "Percentiles":

    st.header("Percentile Analysis")

    col = st.selectbox("Select Column", numeric_cols)

    percentile = st.slider(
        "Select Percentile",
        min_value=0,
        max_value=100,
        value=50
    )

    result = np.percentile(
        df[col].dropna(),
        percentile
    )

    st.metric(f"{percentile}th Percentile", round(result, 2))


# ---------------- HISTOGRAM ----------------

elif menu == "Histogram":

    st.header("Histogram Distribution")

    selected = st.multiselect(
        "Select Columns",
        numeric_cols,
        default=numeric_cols
    )

    for col in selected:

        fig, ax = plt.subplots()

        sb.histplot(df[col].dropna(), kde=True, ax=ax)

        ax.set_title(f"Histogram of {col}")

        st.pyplot(fig)
        plt.close(fig)


# ---------------- SCATTER PLOT ----------------

elif menu == "Scatter Plot":

    st.header("Scatter Plot Analysis")

    col1, col2 = st.columns(2)

    x = col1.selectbox("Select X Axis", numeric_cols)
    y = col2.selectbox(
        "Select Y Axis",
        numeric_cols,
        index=min(1, len(numeric_cols) - 1)
    )

    fig, ax = plt.subplots()

    ax.scatter(df[x], df[y])

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"{x} vs {y}")

    st.pyplot(fig)
    plt.close(fig)


# ---------------- PAIRPLOT ----------------

elif menu == "Pairplot":

    st.header("Pairplot Analysis")

    selected = st.multiselect(
        "Select Columns",
        numeric_cols,
        default=numeric_cols[:4]
    )

    if len(selected) >= 2:

        plot_df = df[selected].dropna()

        if len(plot_df) > 500:
            plot_df = plot_df.sample(500, random_state=42)

        fig = sb.pairplot(plot_df)

        st.pyplot(fig.figure)
        plt.close(fig.figure)

    else:
        st.warning("Please select at least 2 columns.")


# ---------------- SWARMPLOT ----------------

elif menu == "Swarmplot":

    st.header("Swarmplot Analysis")

    col = st.selectbox("Select Column", numeric_cols)

    plot_df = df[[col]].dropna()

    if len(plot_df) > 500:
        plot_df = plot_df.sample(500, random_state=42)

    fig, ax = plt.subplots()

    sb.swarmplot(data=plot_df, x=col, ax=ax)

    ax.set_title(f"Swarmplot of {col}")

    st.pyplot(fig)
    plt.close(fig)


# ---------------- CONFIDENCE INTERVAL ----------------

elif menu == "Confidence Interval":

    st.header("Confidence Interval Calculation")

    sample_type = st.selectbox(
        "Select Sample",
        ["Sample C", "Sample D", "Sample S"]
    )

    c = [
        9.8, 10.3, 10.1, 9.9, 10.4,
        10.2, 9.7, 10.0, 10.5, 9.8,
        10.1, 9.9, 10.2, 10.3, 10.0,
        9.8, 10.4, 10.1, 9.9, 10.2
    ]

    d = [
        29, 31, 30, 32, 28, 29, 31, 30, 29, 32,
        30, 31, 28, 29, 30, 31, 30, 29, 32, 28
    ]

    s = [
        15, 16, 14, 17, 15, 16, 15, 14, 17,
        16, 15, 16, 14, 15, 17, 16, 15
    ]

    if sample_type == "Sample C":
        data = c
        confidence = 0.90

    elif sample_type == "Sample D":
        data = d
        confidence = 0.95

    else:
        data = s
        confidence = 0.99

    mn = np.mean(data)
    sd = np.std(data, ddof=1)

    n = len(data)
    dof = n - 1

    alpha = 1 - confidence

    tscore = t.ppf(1 - alpha / 2, dof)

    moe = tscore * (sd / np.sqrt(n))

    lower = mn - moe
    upper = mn + moe

    st.subheader("Confidence Interval Results")

    st.metric("Sample Mean", round(mn, 4))
    st.metric("Sample Standard Deviation", round(sd, 4))
    st.metric("Sample Size", n)

    st.write("Degrees of Freedom:", dof)
    st.write("T Score:", round(tscore, 4))
    st.write("Margin of Error:", round(moe, 4))

    st.success(
        f"{confidence * 100:.0f}% Confidence Interval: "
        f"({lower:.4f}, {upper:.4f})"
    )


# ---------------- HYPOTHESIS TESTING ----------------

elif menu == "Hypothesis Testing":

    st.header("Hypothesis Testing")

    test = st.selectbox(
        "Select Hypothesis Test",
        ["Test 1", "Test 2"]
    )

    alpha = 0.05

    if test == "Test 1":

        st.subheader("Hypothesis Test 1")

        population_mean = 1.8 + 0.02 * 450
        sample_mean = 10.8
        sample_size = 35
        population_std = 25

    else:

        st.subheader("Hypothesis Test 2")

        population_mean = 10 + 0.08 * 75
        sample_mean = 16.2
        sample_size = 30
        population_std = 6

    standard_error = population_std / np.sqrt(sample_size)

    z_statistic = (
        sample_mean - population_mean
    ) / standard_error

    critical_value = norm.ppf(1 - alpha)

    p_value = 1 - norm.cdf(z_statistic)

    st.write("H0: Sample mean is equal to population mean")
    st.write("H1: Sample mean is greater than population mean")

    st.metric("Population Mean", round(population_mean, 4))
    st.metric("Sample Mean", sample_mean)
    st.metric("Standard Error", round(standard_error, 4))
    st.metric("Z Statistic", round(z_statistic, 4))
    st.metric("Critical Value", round(critical_value, 4))
    st.metric("P Value", round(p_value, 4))

    if z_statistic > critical_value:

        st.success("Reject H0 - Accept H1")

    else:

        st.warning("Fail to Reject H0")

    st.write("Significance Level:", alpha)

