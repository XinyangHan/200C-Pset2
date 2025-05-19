import os
import pandas as pd
import numpy as np
from scipy.stats import zscore
import xgboost
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC

# Paths to data files
FNAME_ADULT_ICU = os.path.join('pset2_data', 'adult_icu.gz')
FNAME_ADULT_NOTES = os.path.join('pset2_data', 'adult_notes.gz')


def load_data():
    """Load the tabular and text datasets."""
    adult_df = pd.read_csv(FNAME_ADULT_ICU, compression='gzip')
    notes_df = pd.read_csv(FNAME_ADULT_NOTES, compression='gzip')
    return adult_df, notes_df


def preprocess_tabular(adult_df):
    """Prepare tabular data and return train/test splits."""
    excl_col = set([
        'subject_id', 'hadm_id', 'icustay_id', 'mort_oneyr', 'mort_hosp',
        'adult_icu', 'train', 'mort_icu', 'valid'
    ])
    binary_col = set([
        'first_hosp_stay', 'first_icu_stay', 'adult_icu',
        'eth_asian', 'eth_black', 'eth_hispanic', 'eth_other', 'eth_white',
        'admType_ELECTIVE', 'admType_EMERGENCY', 'admType_NEWBORN',
        'admType_URGENT'
    ])
    target_col = 'mort_icu'

    feature_col = [i for i in adult_df.columns if i not in excl_col]

    # Z-score non-binary features
    adult_df[feature_col] = adult_df[feature_col].apply(zscore)

    is_train = adult_df['train'] == 1
    is_test = adult_df['train'] == 0

    X = adult_df[feature_col]
    X_train = adult_df[is_train][feature_col]
    y_train = adult_df[is_train][target_col]
    X_test = adult_df[is_test][feature_col]
    y_test = adult_df[is_test][target_col]

    return X, X_train, X_test, y_train, y_test


def train_xgboost(X_train, y_train, X_test, y_test):
    """Train an XGBoost model and return it."""
    d_train = xgboost.DMatrix(X_train, label=y_train)
    d_test = xgboost.DMatrix(X_test, label=y_test)

    params = {
        "eta": 0.01,
        "objective": "binary:logistic",
        "subsample": 0.5,
        "base_score": np.mean(y_train),
        "eval_metric": "logloss",
    }

    model = xgboost.train(
        params,
        d_train,
        5000,
        evals=[(d_test, "test")],
        verbose_eval=100,
        early_stopping_rounds=20,
    )

    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate accuracy of the model on the test set."""
    d_test = xgboost.DMatrix(X_test)
    y_pred_prob = model.predict(d_test)
    y_pred = (y_pred_prob > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")


def shap_tabular(model, X, output_prefix="tabular"):
    """Generate SHAP plots for the tabular model."""
    shap.initjs()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    force_plot = shap.force_plot(
        explainer.expected_value, shap_values[100], X.iloc[100]
    )
    shap.save_html(f"{output_prefix}_force_plot.html", force_plot)

    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_summary.png")
    plt.close()


def preprocess_text(notes_df):
    """Vectorize text data and return train/test splits."""
    feature_col = 'chartext'
    target_col = 'mort_oneyr'

    ctxt = notes_df[feature_col]
    vec = TfidfVectorizer(max_features=500, stop_words='english')
    vec = vec.fit(ctxt)

    is_train = notes_df['train'] == 1
    is_test = notes_df['train'] == 0

    X_train = normalize(vec.transform(notes_df[is_train][feature_col]), norm='l1', axis=0)
    y_train = notes_df[is_train][target_col]

    X_test = normalize(vec.transform(notes_df[is_test][feature_col]), norm='l1', axis=0)
    y_test = notes_df[is_test][target_col]

    return vec, X_train, X_test, y_train, y_test


def shap_text(vec, X_train, y_train, text_to_explain, output_prefix="text"):
    """Generate SHAP explanation for a text sample."""
    model = LinearSVC().fit(X_train, y_train)
    explainer = shap.LinearExplainer(model, X_train)
    x = vec.transform([text_to_explain])
    x_dense = x.toarray()
    shap_values = explainer.shap_values(x_dense)

    force_plot = shap.force_plot(explainer.expected_value, shap_values, text_to_explain.split())
    shap.save_html(f"{output_prefix}_force_plot.html", force_plot)

    feature_num_name_dict = {i: j for i, j in enumerate(vec.get_feature_names_out())}

    print(feature_num_name_dict[200])

    top_increase_idx = np.argsort(shap_values[0])[::-1][:5]
    top_decrease_idx = np.argsort(shap_values[0])[:5]

    increasing = [feature_num_name_dict[i] for i in top_increase_idx]
    decreasing = [feature_num_name_dict[i] for i in top_decrease_idx]

    print("Features increasing ICU mortality:", increasing)
    print("Features decreasing ICU mortality:", decreasing)


def main():
    adult_df, notes_df = load_data()
    X, X_train, X_test, y_train, y_test = preprocess_tabular(adult_df)
    model = train_xgboost(X_train, y_train, X_test, y_test)
    evaluate_model(model, X_test, y_test)
    shap_tabular(model, X)

    vec, X_train_text, X_test_text, y_train_text, y_test_text = preprocess_text(notes_df)
    text_to_explain = "Pt is worsening with liver failure, will transfer to surgery"
    shap_text(vec, X_train_text, y_train_text, text_to_explain)


if __name__ == "__main__":
    main()
