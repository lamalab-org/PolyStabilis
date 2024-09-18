import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
import base64


# Load and preprocess data
@st.cache_data
def load_data():
    data = pd.read_csv(
        "/home/ta45woj/PolySpeckML/data/Ac(e)DexStabilitydata_16_09_cleaned.csv"
    )
    data["tI/2_Acetate"] = pd.to_numeric(data["tI/2_Acetate"], errors="coerce")
    data = data.dropna(subset=["tI/2_Acetate"])
    return data


data = load_data()
X = data[["DS_acycl", "DS_cycl"]]
y = data["tI/2_Acetate"]


# Train the model
@st.cache_resource
def train_model():
    model = GradientBoostingRegressor(
        n_estimators=19, learning_rate=0.16456759118193748, max_depth=5, random_state=42
    )
    model.fit(X, y)
    return model


model = train_model()


# Function to make predictions on a DataFrame
def predict_on_dataframe(df):
    predictions = model.predict(df[["DS_acycl", "DS_cycl"]])
    df["Predicted_tI/2_Acetate"] = predictions
    return df


# Function to create a downloadable CSV
def create_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">Download CSV File</a>'
    return href


# Streamlit app
st.title("PolySpeckML Prediction App")

# Display model information in the sidebar
st.sidebar.header("Model Information")
st.sidebar.write(f"Model Type: Gradient Boosting Regressor")
st.sidebar.write(f"Number of Samples: {len(X)}")
st.sidebar.write(f"Number of Features: {X.shape[1]}")

# Main area for predictions
st.header("Single Prediction")
col1, col2 = st.columns(2)
with col1:
    ds_acycl = st.number_input("Enter DS_acycl:", value=0.5, step=0.01)
with col2:
    ds_cycl = st.number_input("Enter DS_cycl:", value=0.5, step=0.01)

if st.button("Predict"):
    input_data = np.array([[ds_acycl, ds_cycl]])
    prediction = model.predict(input_data)[0]
    st.markdown(f"**Predicted tI/2_Acetate: {prediction:.2f}**")

    # Display 2D contour plot
    fig, ax = plt.subplots()
    x_range = np.linspace(X["DS_acycl"].min(), X["DS_acycl"].max(), 100)
    y_range = np.linspace(X["DS_cycl"].min(), X["DS_cycl"].max(), 100)
    xx, yy = np.meshgrid(x_range, y_range)
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    contour = ax.contourf(xx, yy, Z, levels=20, cmap="viridis", alpha=0.8)
    ax.scatter(X["DS_acycl"], X["DS_cycl"], c=y, cmap="viridis", edgecolor="black")
    ax.scatter(ds_acycl, ds_cycl, color="red", s=200, marker="*", label="Prediction")
    ax.set_xlabel("DS_acycl")
    ax.set_ylabel("DS_cycl")
    ax.set_title("tI/2_Acetate Prediction Contour Plot")
    plt.colorbar(contour, label="Predicted tI/2_Acetate")
    ax.legend()
    st.pyplot(fig)

# CSV Upload and Prediction
st.header("Batch Prediction from CSV")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)

    if "DS_acycl" in input_df.columns and "DS_cycl" in input_df.columns:
        st.write("Preview of uploaded data:")
        st.write(input_df.head())

        if st.button("Make Batch Predictions"):
            output_df = predict_on_dataframe(input_df)
            st.write("Preview of predictions:")
            st.write(output_df.head())

            # Create download link
            csv = output_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">Download Predictions CSV</a>'
            st.markdown(href, unsafe_allow_html=True)
    else:
        st.error("The uploaded CSV must contain 'DS_acycl' and 'DS_cycl' columns.")

# Display data statistics
st.header("Data Statistics")
st.write(data[["DS_acycl", "DS_cycl", "tI/2_Acetate"]].describe())

# Allow users to view raw data
if st.checkbox("Show training data"):
    st.write(data)
