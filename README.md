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
streamlit polyspeck_app.py
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

```
MIT License

Copyright (c) [year] [fullname]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
