# PolySpeckML: Polymer Nanoparticle Stability Prediction

PolySpeckML is a machine learning project aimed at predicting the stability of polymer nanoparticles based on their composition and properties. This project uses data on polymer nanoparticles, including their acyclic and cyclic Degree of Substitution (DS), to predict stability measurements in different buffer conditions.

## Project Overview

The main goals of this project are:

1. Predict nanoparticle stability in Acetate and PBS buffers using polymer properties.
2. Analyze and preprocess provided data on polymer nanoparticles.
3. Develop and evaluate machine learning models for stability prediction.
4. Create a user-friendly interface for making predictions.

## Getting Started

### Prerequisites

- Python 3.10
- pip (Python package installer)

### Installation

1. Clone the repository:

   ```
   git clone https://github.com/yourusername/PolySpeckML.git
   cd PolySpeckML
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Project Structure

- `data/`: Contains the dataset used for training and testing.
- `notebooks/`: Jupyter notebooks for data analysis and model development.
- `polyspeck_app.py`: Streamlit application for the user interface.
- `requirements.txt`: List of Python dependencies.

## Usage

### Data Analysis and Model Training

To explore the data and train the models, run the Jupyter notebooks in the `notebooks/` directory.

### Running the Streamlit App

To launch the prediction interface, run:

```
streamlit run polyspeck_app.py -- --data_path="/path/to/your/data.csv"
```

This will start a local server, and you can access the app through your web browser.

## Features

- Data preprocessing and analysis
- Machine learning model development (Gradient Boosting Regressor, Logistic Regression)
- Cross-validation techniques (Leave-One-Out Cross-Validation)
- Interactive web application for single and batch predictions
- Visualization of prediction results

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
