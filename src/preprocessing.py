import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

ADDONS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
          "TechSupport", "StreamingTV", "StreamingMovies"]

NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges",
           "n_addons", "avg_monthly_spend", "charge_gap"]

BINARY_FLAGS = ["SeniorCitizen", "is_autopay", "no_protection", "monthly_fiber"]

CATEGORICAL = ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
               "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
               "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
               "PaperlessBilling", "PaymentMethod", "tenure_group"]


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw Telco dataset."""
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    assert (df.loc[df["TotalCharges"].isna(), "tenure"] == 0).all(), \
        "Unexpected missing TotalCharges for customers with tenure > 0"
    df["TotalCharges"] = df["TotalCharges"].fillna(0)
    df = df.drop(columns="customerID", errors="ignore") 
    df = df.replace({"No internet service": "No", "No phone service": "No"})
    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create new features. All of them are row-wise, so they cannot leak
    information between train and test."""
    df = df.copy()

    # How many add-on services the customer subscribes to (0 to 6)
    df["n_addons"] = (df[ADDONS] == "Yes").sum(axis=1)

    # Average amount actually paid per month since signup
    # (new customers with tenure 0: use their current monthly charge)
    df["avg_monthly_spend"] = np.where(
        df["tenure"] > 0, df["TotalCharges"] / df["tenure"].clip(lower=1),
        df["MonthlyCharges"])

    # Positive = customer pays more now than on average in the past (price increase / upsell)
    df["charge_gap"] = df["MonthlyCharges"] - df["avg_monthly_spend"]

    # Lifecycle stage
    df["tenure_group"] = pd.cut(df["tenure"], bins=[-1, 12, 24, 48, 72],
                                labels=["0-12", "13-24", "25-48", "49-72"]).astype(str)

    # Automatic payment (bank transfer or credit card)
    df["is_autopay"] = df["PaymentMethod"].str.contains("automatic").astype(int)

    # Has internet but neither security nor tech support
    df["no_protection"] = ((df["InternetService"] != "No")
                           & (df["OnlineSecurity"] == "No")
                           & (df["TechSupport"] == "No")).astype(int)

    # Interaction between the two strongest EDA signals
    df["monthly_fiber"] = ((df["Contract"] == "Month-to-month")
                           & (df["InternetService"] == "Fiber optic")).astype(int)
    return df


def build_preprocessor() -> ColumnTransformer:
    """Fresh (unfitted) preprocessing transformer."""
    prep = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("bin", "passthrough", BINARY_FLAGS),
        ("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore",
                              sparse_output=False), CATEGORICAL),
    ])
    prep.set_output(transform="pandas")   
    return prep