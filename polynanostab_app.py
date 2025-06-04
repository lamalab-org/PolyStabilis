import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import LeaveOneOut
import base64
import fire
import joblib
import os
import hashlib


def get_file_hash(file_path):
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def run_app(data_path):
    # Load and preprocess data with hash check
    @st.cache_data(show_spinner=True)
    def load_data(file_path, file_hash):
        data = pd.read_csv(file_path)
        data["tI/2_Acetate"] = pd.to_numeric(data["tI/2_Acetate"], errors="coerce")
        data = data.dropna(subset=["tI/2_Acetate"])
        return data

    # Get current file hash
    current_hash = get_file_hash(data_path)

    # Load data with current hash
    data = load_data(data_path, current_hash)
    X = data[["DS_acycl", "DS_cycl"]]
    y_acetate = data["tI/2_Acetate"]

    # Encode the 'tI/2_PBS' column
    le = LabelEncoder()
    y_pbs = le.fit_transform(data["tI/2_PBS"])

    # Train and store regression models
    @st.cache_resource(show_spinner=True)
    def train_and_store_regression_models(X, y_acetate, current_hash):
        loo = LeaveOneOut()
        models_dir = "regression_models"

        if os.path.exists(models_dir):
            for file in os.listdir(models_dir):
                os.remove(os.path.join(models_dir, file))
        else:
            os.makedirs(models_dir)

        models = []
        for i, (train_index, test_index) in enumerate(loo.split(X)):
            X_train = X.iloc[train_index]
            y_train = y_acetate.iloc[train_index]

            model = GradientBoostingRegressor(
                n_estimators=19,
                learning_rate=0.16456759118193748,
                max_depth=5,
                random_state=42,
            )
            model.fit(X_train, y_train)
            models.append(model)

            model_filename = f"{models_dir}/model_sample{i}.joblib"
            joblib.dump(model, model_filename)

        return models

    # Train the classification model
    @st.cache_resource(show_spinner=True)
    def train_classification_model(X, y_pbs, current_hash):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = LogisticRegression(class_weight="balanced", random_state=42)
        model.fit(X_scaled, y_pbs)
        return model, scaler

    # Train models
    regression_models = train_and_store_regression_models(X, y_acetate, current_hash)
    classification_model, scaler = train_classification_model(X, y_pbs, current_hash)

    def create_stability_plots(input_point=None):
        """Create both stability plots with consistent styling"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Create mesh grid
        x_range = np.linspace(0.1, 1.3, 100)
        y_range = np.linspace(0.5, 2.1, 100)  
        xx, yy = np.meshgrid(x_range, y_range)

        # Acetate Stability Plot (Left)
        Z_acetate = np.mean(
            [
                model.predict(np.c_[xx.ravel(), yy.ravel()])
                for model in regression_models
            ],
            axis=0,
        ).reshape(xx.shape)

        # Create discrete levels for acetate plot
        levels = np.arange(0, 25, 1)
        contour1 = ax1.contourf(xx, yy, Z_acetate, levels=levels, cmap="viridis")
        scatter1 = ax1.scatter(
            X["DS_acycl"], X["DS_cycl"], c="#EB8317", edgecolor="black", s=50
        )

        if input_point is not None:
            ax1.scatter(
                input_point[0],
                input_point[1],
                color="red",
                s=200,
                marker="*",
                label="Input Point",
            )

        ax1.set_xlabel("DS_acycl")
        ax1.set_ylabel("DS_cycl")
        plt.colorbar(
            contour1,
            ax=ax1,
            label="Predicted tI/2 Acetate Stability",
            ticks=np.arange(0, 25, 2),
        )

        # PBS Stability Plot (Right)
        X_mesh = np.c_[xx.ravel(), yy.ravel()]
        Z_pbs = classification_model.predict(scaler.transform(X_mesh)).reshape(xx.shape)

        contour2 = ax2.contourf(
            xx, yy, Z_pbs, levels=1, colors=["#fde725", "#440154"], alpha=0.8
        )
        ax2.scatter(X["DS_acycl"], X["DS_cycl"], c="#EB8317", edgecolor="black", s=50)

        if input_point is not None:
            ax2.scatter(
                input_point[0],
                input_point[1],
                color="red",
                s=200,
                marker="*",
                label="Input Point",
            )

        ax2.set_xlabel("DS_acycl")
        ax2.set_ylabel("DS_cycl")
        cbar2 = plt.colorbar(contour2, ax=ax2, label="Predicted tI/2 PBS Stability")

        for ax in [ax1, ax2]:
            ax.set_xlim(0.1, 1.3)
            ax.set_ylim(0.5, 2.1)
            if input_point is not None:
                ax.legend()

        plt.tight_layout()
        return fig

    def predict_on_dataframe(df):
        """Make predictions for a batch of inputs"""
        acetate_predictions = []
        acetate_stds = []

        for _, row in df.iterrows():
            input_data = row[["DS_acycl", "DS_cycl"]].values.reshape(1, -1)
            predictions = [model.predict(input_data)[0] for model in regression_models]
            acetate_predictions.append(np.mean(predictions))
            acetate_stds.append(np.std(predictions))

        df["Predicted_tI/2_Acetate"] = acetate_predictions
        df["Predicted_tI/2_Acetate_Std"] = acetate_stds

        X_scaled = scaler.transform(df[["DS_acycl", "DS_cycl"]])
        pbs_predictions = classification_model.predict(X_scaled)
        df["Predicted_Stability_PBS"] = le.inverse_transform(pbs_predictions)

        return df

    st.title("Polymer Nanoparticle Stability Prediction App")

    # Model information in sidebar
    st.sidebar.header("Model Information")
    st.sidebar.write("Regression Model: Ensemble of Gradient Boosting Regressors")
    st.sidebar.write("Classification Model: Logistic Regression")
    st.sidebar.write(f"Number of Samples: {len(X)}")
    st.sidebar.write(f"Number of Features: {X.shape[1]}")

    # Single prediction section
    st.header("Single Prediction of Nanoparticle Stability")
    col1, col2 = st.columns(2)
    with col1:
        ds_acycl = st.number_input("Enter DS_acycl:", value=0.5, step=0.01)
    with col2:
        ds_cycl = st.number_input("Enter DS_cycl:", value=0.5, step=0.01)

    if st.button("Predict Stability"):
        input_data = np.array([[ds_acycl, ds_cycl]])

        # Make predictions
        acetate_predictions = [
            model.predict(input_data)[0] for model in regression_models
        ]
        acetate_prediction = np.mean(acetate_predictions)
        acetate_std = np.std(acetate_predictions)

        pbs_prediction = classification_model.predict(scaler.transform(input_data))[0]
        pbs_stability = le.inverse_transform([pbs_prediction])[0]

        # Display predictions
        st.markdown(
            f"**Predicted Nanoparticle Stability (t1/2 Acetate): {acetate_prediction:.2f} ± {acetate_std:.2f}**"
        )
        st.markdown(f"**Predicted Nanoparticle Stability (t1/2 PBS): {pbs_stability}**")

        # Show plots with input point
        fig = create_stability_plots(input_point=[ds_acycl, ds_cycl])
        st.pyplot(fig)
    else:
        # Show plots without input point
        fig = create_stability_plots()
        st.pyplot(fig)

    # CSV Upload and Batch Prediction
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
                href = f'<a href="data:file/csv;base64,{b64}" download="stability_predictions.csv">Download Predictions CSV</a>'
                st.markdown(href, unsafe_allow_html=True)
        else:
            st.error("CSV must contain 'DS_acycl' and 'DS_cycl' columns.")

    # Data Statistics Section
    st.header("Data Statistics")

    # Numeric columns
    numeric_cols = ["DS_acycl", "DS_cycl", "tI/2_Acetate"]
    st.subheader("Numeric Columns Statistics")
    st.write(data[numeric_cols].describe())

    # Categorical column
    st.subheader("PBS Stability Distribution")
    pbs_stats = data["tI/2_PBS"].value_counts(normalize=True).reset_index()
    pbs_stats.columns = ["Category", "Percentage"]
    pbs_stats["Percentage"] = pbs_stats["Percentage"] * 100
    st.write(pbs_stats)

    # Show raw data option
    if st.checkbox("Show Raw Training Data"):
        st.write(data)


if __name__ == "__main__":
    fire.Fire(run_app)
