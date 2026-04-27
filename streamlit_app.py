import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


st.set_page_config(
    page_title="Adult Income Prediction Dashboard",
    layout="wide"
)

st.title("Adult Income Prediction Dashboard")
st.caption("Predictive model only: feature importance and predictions should not be interpreted as causal effects.")


@st.cache_data
def load_data():
    columns = [
        "age", "workclass", "fnlwgt", "education", "education_num",
        "marital_status", "occupation", "relationship", "race", "sex",
        "capital_gain", "capital_loss", "hours_per_week",
        "native_country", "income"
    ]

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

    df = pd.read_csv(
        url,
        header=None,
        names=columns,
        na_values="?",
        skipinitialspace=True
    )

    return df


@st.cache_resource
def train_model(df):
    X = df.drop(columns=["income"])
    y = df["income"]

    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_features = X.select_dtypes(exclude=["object"]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=250,
            max_depth=12,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1
        ))
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    positive_index = list(model.named_steps["classifier"].classes_).index(">50K")
    y_prob = model.predict_proba(X_test)[:, positive_index]

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, pos_label=">50K"),
        "Recall": recall_score(y_test, y_pred, pos_label=">50K"),
        "F1": f1_score(y_test, y_pred, pos_label=">50K"),
        "ROC AUC": roc_auc_score((y_test == ">50K").astype(int), y_prob)
    }

    return model, metrics


def get_tree_probability_interval(model, user_input):
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    transformed_input = preprocessor.transform(user_input)
    class_labels = list(classifier.classes_)
    positive_index = class_labels.index(">50K")

    tree_probs = np.array([
        tree.predict_proba(transformed_input)[0, positive_index]
        for tree in classifier.estimators_
    ])

    point_estimate = classifier.predict_proba(transformed_input)[0, positive_index]
    lower_bound = np.percentile(tree_probs, 2.5)
    upper_bound = np.percentile(tree_probs, 97.5)

    return point_estimate, lower_bound, upper_bound


df = load_data()
model, metrics = train_model(df)

st.sidebar.header("Model Inputs")

age = st.sidebar.slider("Age", 17, 90, 35)
workclass = st.sidebar.selectbox("Workclass", sorted(df["workclass"].dropna().unique()))
education = st.sidebar.selectbox("Education", sorted(df["education"].dropna().unique()))
education_num = int(df.loc[df["education"] == education, "education_num"].median())
marital_status = st.sidebar.selectbox("Marital Status", sorted(df["marital_status"].dropna().unique()))
occupation = st.sidebar.selectbox("Occupation", sorted(df["occupation"].dropna().unique()))
relationship = st.sidebar.selectbox("Relationship", sorted(df["relationship"].dropna().unique()))
race = st.sidebar.selectbox("Race", sorted(df["race"].dropna().unique()))
sex = st.sidebar.selectbox("Sex", sorted(df["sex"].dropna().unique()))
capital_gain = st.sidebar.number_input("Capital Gain", min_value=0, max_value=100000, value=0, step=500)
capital_loss = st.sidebar.number_input("Capital Loss", min_value=0, max_value=10000, value=0, step=100)
hours_per_week = st.sidebar.slider("Hours per Week", 1, 99, 40)
native_country = st.sidebar.selectbox("Native Country", sorted(df["native_country"].dropna().unique()))

fnlwgt = int(df["fnlwgt"].median())

user_input = pd.DataFrame([{
    "age": age,
    "workclass": workclass,
    "fnlwgt": fnlwgt,
    "education": education,
    "education_num": education_num,
    "marital_status": marital_status,
    "occupation": occupation,
    "relationship": relationship,
    "race": race,
    "sex": sex,
    "capital_gain": capital_gain,
    "capital_loss": capital_loss,
    "hours_per_week": hours_per_week,
    "native_country": native_country
}])

point_estimate, lower_bound, upper_bound = get_tree_probability_interval(model, user_input)
predicted_class = ">50K" if point_estimate >= 0.50 else "<=50K"

left, middle, right = st.columns(3)

with left:
    st.metric("Predicted Class", predicted_class)

with middle:
    st.metric("Predicted Probability of >50K", f"{point_estimate:.1%}")

with right:
    st.metric("Approx. 95% Prediction Interval", f"{lower_bound:.1%} to {upper_bound:.1%}")

st.write(
    "The interval is estimated from variation across the random forest's individual trees. "
    "It is a rough prediction uncertainty range, not a causal confidence interval."
)

st.subheader("Selected Input Profile")
st.dataframe(user_input, use_container_width=True)

st.subheader("Model Performance on Held-Out Test Set")

metric_cols = st.columns(len(metrics))
for col, (metric_name, metric_value) in zip(metric_cols, metrics.items()):
    col.metric(metric_name, f"{metric_value:.3f}")

st.subheader("Interactive Visualization")

viz_option = st.selectbox(
    "Choose a visualization",
    [
        "Income rate by education",
        "Income rate by workclass",
        "Hours per week by income class",
        "Age distribution by income class"
    ]
)

if viz_option == "Income rate by education":
    chart_df = (
        df.groupby("education", as_index=False)
        .agg(income_rate_gt_50k=("income", lambda x: (x == ">50K").mean()),
             count=("income", "size"))
        .sort_values("income_rate_gt_50k", ascending=False)
    )

    chart = alt.Chart(chart_df).mark_bar().encode(
        x=alt.X("education:N", sort="-y", title="Education"),
        y=alt.Y("income_rate_gt_50k:Q", title="Share earning >50K", axis=alt.Axis(format="%")),
        tooltip=["education", alt.Tooltip("income_rate_gt_50k:Q", format=".1%"), "count"]
    ).properties(height=420)

    st.altair_chart(chart, use_container_width=True)

elif viz_option == "Income rate by workclass":
    chart_df = (
        df.groupby("workclass", as_index=False)
        .agg(income_rate_gt_50k=("income", lambda x: (x == ">50K").mean()),
             count=("income", "size"))
        .dropna()
        .sort_values("income_rate_gt_50k", ascending=False)
    )

    chart = alt.Chart(chart_df).mark_bar().encode(
        x=alt.X("workclass:N", sort="-y", title="Workclass"),
        y=alt.Y("income_rate_gt_50k:Q", title="Share earning >50K", axis=alt.Axis(format="%")),
        tooltip=["workclass", alt.Tooltip("income_rate_gt_50k:Q", format=".1%"), "count"]
    ).properties(height=420)

    st.altair_chart(chart, use_container_width=True)

elif viz_option == "Hours per week by income class":
    chart = alt.Chart(df.sample(min(5000, len(df)), random_state=42)).mark_circle(opacity=0.35).encode(
        x=alt.X("hours_per_week:Q", title="Hours per week"),
        y=alt.Y("age:Q", title="Age"),
        color=alt.Color("income:N", title="Income class"),
        tooltip=["age", "hours_per_week", "education", "occupation", "income"]
    ).properties(height=420).interactive()

    st.altair_chart(chart, use_container_width=True)

else:
    chart = alt.Chart(df.sample(min(5000, len(df)), random_state=42)).mark_bar(opacity=0.75).encode(
        x=alt.X("age:Q", bin=alt.Bin(maxbins=30), title="Age"),
        y=alt.Y("count():Q", title="Count"),
        color=alt.Color("income:N", title="Income class"),
        tooltip=["income", "count()"]
    ).properties(height=420).interactive()

    st.altair_chart(chart, use_container_width=True)

st.subheader("Interpretation")
st.write(
    "Use the controls in the sidebar to test how the model changes its prediction for different profiles. "
    "The dashboard is designed for prediction, not causal explanation. For example, if education or hours worked "
    "is associated with a higher predicted probability, that does not prove that changing that variable alone would cause income to increase."
)
