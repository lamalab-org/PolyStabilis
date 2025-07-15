# PolyStabilis: Polymer Nanoparticle Stability Prediction

PolyNanoStabML is a machine learning-based application for predicting the stability of polymer nanoparticles. The tool leverages regression and classification models to predict nanoparticle stability. Specifically, it uses regression model to predict half-life stability in acetate buffer and classification models to predict stability in PBS buffer.

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
   git clone https://github.com/lamalab-org/PolyNanoStabML.git
   cd PolyNanoStabML
   ```

2. Set up the environment:

   ```
   conda create --name polynanostab python=3.10
   conda activate polynanostab
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Project Structure

- `data/`: Contains the dataset used for training and testing.
- `notebooks/`: Jupyter notebooks for data analysis and model development.
- `polynanostab_app.py`: Streamlit application for the user interface.
- `requirements.txt`: List of Python dependencies.

## Usage

### Data Analysis and Model Training

To explore the data and train the models, run the Jupyter notebooks in the `notebooks/` directory.

### Running the Streamlit App

To launch the prediction interface, run:

```
streamlit run polystabilis_app.py -- --data_path="/path/to/your/data.csv"
```

This will start a local server, and you can access the app through your web browser.


## App link

You can access the deployed application at [PolyStabilis App](https://polystabilis.streamlit.app).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
