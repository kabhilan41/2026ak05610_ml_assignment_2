import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                            recall_score, f1_score, matthews_corrcoef,
                            confusion_matrix, classification_report)

TARGET = "Revenue"

# model file and the preprocessor file used while training that model
MODELS = {
    "Logistic Regression": ("model/logistic_regression.joblib", "model/preprocessor_scaled.joblib"),
    "Decision Tree": ("model/decision_tree.joblib", "model/preprocessor_plain.joblib"),
    "KNN": ("model/knn.joblib", "model/preprocessor_scaled.joblib"),
    "Naive Bayes": ("model/naive_bayes.joblib", "model/preprocessor_plain.joblib"),
    "Support Vector Machine": ("model/svm.joblib", "model/preprocessor_scaled.joblib"),
}

METRIC_NAMES = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


@st.cache_resource
def load_file(path):
    return joblib.load(path)


def get_scores(model, X):
    # SVM (rbf) has no predict_proba, so decision_function is used for the AUC
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def calculate_metrics(model, preprocessor, data):
    X = preprocessor.transform(data.drop(columns=[TARGET]))
    y_true = data[TARGET]
    y_pred = model.predict(X)
    y_score = get_scores(model, X)

    scores = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_score),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
    return scores, y_true, y_pred


def draw_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["No purchase", "Purchase"])
    ax.set_yticks([0, 1], ["No purchase", "Purchase"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=12)
    fig.tight_layout()
    return fig


st.set_page_config(page_title="Online Shopper Purchase Prediction", page_icon="🛒", layout="wide")

st.title("Online Shopper Purchase Prediction")
st.write("Machine Learning Assignment 2 | Kabhilan S | Roll No: 2026AK05610")
st.write(
    "This app loads the 5 trained classification models saved in the `model` folder and checks "
    "them on a test data file. The models are **not** trained again here."
)

with st.sidebar:
    st.header("Inputs")
    selected_model = st.selectbox("Select classification model", list(MODELS.keys()))
    uploaded_file = st.file_uploader("Upload test data (.csv)", type=["csv"])
    st.caption("If no file is uploaded, the test_data.csv of this repository is used.")

if uploaded_file is None:
    data = pd.read_csv("test_data.csv")
    st.info("Using the default test_data.csv of the repository.")
else:
    data = pd.read_csv(uploaded_file)
    st.info("Using the uploaded file: " + uploaded_file.name)

if TARGET not in data.columns:
    st.error("The uploaded file must contain the target column '" + TARGET + "'.")
    st.stop()

# in the raw csv Weekend and Revenue are True/False, the models were trained on 0/1
data = data.replace({"True": 1, "False": 0})
for column in ["Weekend", TARGET]:
    if column in data.columns and data[column].dtype == bool:
        data[column] = data[column].astype(int)

st.subheader("Test data")
col1, col2, col3 = st.columns(3)
col1.metric("Rows", data.shape[0])
col2.metric("Columns", data.shape[1])
col3.metric("Purchase sessions", int(data[TARGET].sum()))
st.dataframe(data.head(5))

model_path, preprocessor_path = MODELS[selected_model]
model = load_file(model_path)
preprocessor = load_file(preprocessor_path)

scores, y_true, y_pred = calculate_metrics(model, preprocessor, data)

st.subheader("Evaluation metrics - " + selected_model)
metric_columns = st.columns(6)
for column, name in zip(metric_columns, METRIC_NAMES):
    column.metric(name, round(scores[name], 4))

left, right = st.columns([1, 1.4])
with left:
    st.write("**Confusion matrix**")
    st.pyplot(draw_confusion_matrix(y_true, y_pred))

with right:
    st.write("**Classification report**")
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose().round(3)
    st.dataframe(report_df)

st.subheader("All 5 models on this test data")
comparison = []
for name, (model_path, preprocessor_path) in MODELS.items():
    model_scores, _, _ = calculate_metrics(load_file(model_path), load_file(preprocessor_path), data)
    model_scores["ML Model Name"] = name
    comparison.append(model_scores)

comparison_df = pd.DataFrame(comparison).set_index("ML Model Name")[METRIC_NAMES].round(4)
st.dataframe(comparison_df)

st.write("**Accuracy, F1 and MCC of all the models**")
st.bar_chart(comparison_df[["Accuracy", "F1", "MCC"]])

st.caption(
    "Dataset: Online Shoppers Purchasing Intention (UCI). "
    "Target column: Revenue (True = the session ended in a purchase)."
)
