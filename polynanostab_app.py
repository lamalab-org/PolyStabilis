import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
import base64
import fire


def run_app(data_path):
    # Load and preprocess data
    @st.cache_data
    def load_data():
        data = pd.read_csv(data_path)
        data["tI/2_Acetate"] = pd.to_numeric(data["tI/2_Acetate"], errors="coerce")
        data = data.dropna(subset=["tI/2_Acetate"])
        return data

    data = load_data()
    X = data[["DS_acycl", "DS_cycl"]]
    y_acetate = data["tI/2_Acetate"]

    # Encode the 'tI/2_PBS' column
    le = LabelEncoder()
    y_pbs = le.fit_transform(data["tI/2_PBS"])

    # Train the regression model
    @st.cache_resource
    def train_regression_model():
        model = GradientBoostingRegressor(
            n_estimators=19,
            learning_rate=0.16456759118193748,
            max_depth=5,
            random_state=42,
        )
        model.fit(X, y_acetate)
        return model

    # Train the classification model
    @st.cache_resource
    def train_classification_model():
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = LogisticRegression(class_weight="balanced", random_state=42)
        model.fit(X_scaled, y_pbs)
        return model, scaler

    regression_model = train_regression_model()
    classification_model, scaler = train_classification_model()

    # Function to make predictions on a DataFrame
    def predict_on_dataframe(df):
        df["Predicted_tI/2_Acetate"] = regression_model.predict(
            df[["DS_acycl", "DS_cycl"]]
        )
        X_scaled = scaler.transform(df[["DS_acycl", "DS_cycl"]])
        pbs_predictions = classification_model.predict(X_scaled)
        df["Predicted_Stability_PBS"] = le.inverse_transform(pbs_predictions)
        return df

    # Streamlit app
    st.title("PolySpeckML Nanoparticle Stability Prediction App")

    # Display model information in the sidebar
    st.sidebar.header("Model Information")
    st.sidebar.write("Regression Model: Gradient Boosting Regressor")
    st.sidebar.write("Classification Model: Logistic Regression")
    st.sidebar.write(f"Number of Samples: {len(X)}")
    st.sidebar.write(f"Number of Features: {X.shape[1]}")

    # Main area for predictions
    st.header("Single Prediction of Nanoparticle Stability")
    st.write(
        "Predict the stability of a nanoparticle in terms of t1/2 Acetate and t1/2 PBS."
    )
    col1, col2 = st.columns(2)
    with col1:
        ds_acycl = st.number_input("Enter DS_acycl:", value=0.5, step=0.01)
    with col2:
        ds_cycl = st.number_input("Enter DS_cycl:", value=0.5, step=0.01)

    if st.button("Predict Stability"):
        input_data = np.array([[ds_acycl, ds_cycl]])
        acetate_prediction = regression_model.predict(input_data)[0]
        pbs_prediction = classification_model.predict(scaler.transform(input_data))[0]
        pbs_stability = le.inverse_transform([pbs_prediction])[0]

        st.markdown(
            f"**Predicted Nanoparticle Stability (t1/2 Acetate): {acetate_prediction:.2f}**"
        )
        st.markdown(f"**Predicted Nanoparticle Stability (t1/2 PBS): {pbs_stability}**")

        # Display 2D contour plot for t1/2 Acetate and t1/2 PBS
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        x_range = np.linspace(X["DS_acycl"].min(), X["DS_acycl"].max(), 100)
        y_range = np.linspace(X["DS_cycl"].min(), X["DS_cycl"].max(), 100)
        xx, yy = np.meshgrid(x_range, y_range)

        # t1/2 Acetate plot
        Z_acetate = regression_model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(
            xx.shape
        )
        contour1 = ax1.contourf(xx, yy, Z_acetate, levels=20, cmap="viridis", alpha=0.8)
        ax1.scatter(
            X["DS_acycl"], X["DS_cycl"], c=y_acetate, cmap="viridis", edgecolor="black"
        )
        ax1.scatter(
            ds_acycl, ds_cycl, color="red", s=200, marker="*", label="Input Point"
        )
        ax1.set_xlabel("DS_acycl")
        ax1.set_ylabel("DS_cycl")
        ax1.set_title("t1/2 Acetate Prediction")
        plt.colorbar(contour1, ax=ax1, label="Predicted t1/2 Acetate")
        ax1.legend()

        # t1/2 PBS plot
        Z_pbs = classification_model.predict(
            scaler.transform(np.c_[xx.ravel(), yy.ravel()])
        ).reshape(xx.shape)
        contour2 = ax2.contourf(xx, yy, Z_pbs, levels=1, cmap="coolwarm", alpha=0.8)
        ax2.scatter(
            X["DS_acycl"], X["DS_cycl"], c=y_pbs, cmap="coolwarm", edgecolor="black"
        )
        ax2.scatter(
            ds_acycl, ds_cycl, color="red", s=200, marker="*", label="Input Point"
        )
        ax2.set_xlabel("DS_acycl")
        ax2.set_ylabel("DS_cycl")
        ax2.set_title("t1/2 PBS Stability Prediction")
        plt.colorbar(
            contour2, ax=ax2, label="Predicted Stability (0: Unstable, 1: Stable)"
        )
        ax2.legend()

        st.pyplot(fig)

    # CSV Upload and Prediction
    st.header("Batch Prediction of Nanoparticle Stability from CSV")
    st.write("Upload a CSV file to predict the stability of multiple nanoparticles.")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
        if "DS_acycl" in input_df.columns and "DS_cycl" in input_df.columns:
            st.write("Preview of uploaded data:")
            st.write(input_df.head())
            if st.button("Make Batch Stability Predictions"):
                output_df = predict_on_dataframe(input_df)
                st.write("Preview of nanoparticle stability predictions:")
                st.write(output_df.head())
                # Create download link
                csv = output_df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="nanoparticle_stability_predictions.csv">Download Nanoparticle Stability Predictions CSV</a>'
                st.markdown(href, unsafe_allow_html=True)
        else:
            st.error("The uploaded CSV must contain 'DS_acycl' and 'DS_cycl' columns.")

    # Display data statistics
    st.header("Data Statistics")

    # Numeric columns
    numeric_cols = ["DS_acycl", "DS_cycl", "tI/2_Acetate"]
    st.subheader("Numeric Columns Statistics")
    st.write(data[numeric_cols].describe())

    # Categorical column (tI/2_PBS)
    st.subheader("Categorical Column Statistics (tI/2_PBS)")
    pbs_stats = data["tI/2_PBS"].value_counts(normalize=True).reset_index()
    pbs_stats.columns = ["Category", "Percentage"]
    pbs_stats["Percentage"] = pbs_stats["Percentage"] * 100
    st.write(pbs_stats)

    # Visualization for tI/2_PBS distribution
    st.subheader("Distribution of tI/2_PBS")
    fig, ax = plt.subplots()
    data["tI/2_PBS"].value_counts().plot(kind="bar", ax=ax)
    ax.set_xlabel("Category")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    # Allow users to view raw data
    if st.checkbox("Show training data"):
        st.write(data)


if __name__ == "__main__":
    fire.Fire(run_app)
