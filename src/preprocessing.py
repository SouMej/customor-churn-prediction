import pandas as pd


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw Telco dataset."""
    df = df.copy()

    # TotalCharges: blank strings -> NaN -> 0 (only for tenure == 0 customers)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    assert (df.loc[df["TotalCharges"].isna(), "tenure"] == 0).all(), \
        "Unexpected missing TotalCharges for customers with tenure > 0"
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # Drop the identifier
    df = df.drop(columns="customerID")

    # Merge redundant categories
    df = df.replace({"No internet service": "No", "No phone service": "No"})

    # Encode the target
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    return df