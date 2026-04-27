# Adult Income Prediction Dashboard

This project predicts whether a person's income is likely to be above or below $50K using the UCI Adult Income dataset. The goal is not to explain what causes income differences, but to build and compare machine learning models for a real prediction task.

## Project Files

- `app.py` - Streamlit dashboard
- `requirements.txt` - Python packages needed to run the app
- `econ3916_ml_prediction_project_viswanathan_akash.ipynb` - full analysis notebook
- `README.md` - setup and reproducibility instructions

## Dataset

The project uses the UCI Adult Income dataset.

Dataset source: https://archive.ics.uci.edu/dataset/2/adult

The app loads the data directly from the UCI public URL, so no separate data file is required to run the dashboard.

## How to Run Locally

Clone the repository:

```bash
git clone https://github.com/akash-v888/ECON3916_Final_Project.git
cd ECON3916_Final_Project
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run streamlit_app.py
```

The dashboard should open in your browser at:

```text
http://localhost:8501
```

## Streamlit Dashboard

The dashboard lets users adjust input features such as age, education, occupation, workclass, capital gain, and hours worked per week. It then returns:

- predicted income class
- predicted probability of earning more than $50K
- approximate prediction interval
- interactive visualizations based on the dataset

The dashboard is meant for prediction only. The model should not be interpreted as proving that any feature causes higher or lower income.

## Reproducing the Analysis

To reproduce the full analysis:

1. Open the Jupyter notebook.
2. Run all cells from top to bottom.
3. Confirm that the dataset loads from the UCI URL.
4. Check the model comparison results and visualizations.
5. Run `streamlit run streamlit_app.py` to test the dashboard locally.
