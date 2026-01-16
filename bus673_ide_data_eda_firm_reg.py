"""
Prereqs (run in terminal before this script):

    python -m venv venv
    source venv/bin/activate        # macOS/Linux
    # venv\\Scripts\\activate       # Windows

    pip3 install google-cloud-bigquery pandas numpy linearmodels matplotlib db-dtypes
    pip3 install google-cloud-bigquery-storage scikit-learn statsmodels
    # pip install db-dtypes

    gcloud auth login
    gcloud config set project sharp-footing-478019-d7
    gcloud auth application-default login

    # run the script
    # first make sure you are in the folder where you save your file 
    python3 bus673_ide_data_eda_firm_reg.py
"""


# -------------------------
# IMPORTS
# -------------------------
from google.cloud import bigquery
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans  # not used yet, but kept for later exercises
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
import statsmodels.api as sm


# -------------------------
# CONFIGURATION
# Remember to replace them with your project ID, dataset name, and table name 
# -------------------------
PROJECT_ID = "bus673"
DATASET = "bus673_compustat"
TABLE = "annual_data2000_2025"

FULL_TABLE_NAME = f"`{PROJECT_ID}.{DATASET}.{TABLE}`"
# ROW_LIMIT = 20000  # optional if you later want to limit rows


# -------------------------
# BIGQUERY CLIENT SETUP
# -------------------------
def get_bq_client():
    """Create and return a BigQuery client."""
    return bigquery.Client(project=PROJECT_ID)


# -------------------------
# LOAD DATA
# -------------------------
def load_compustat_data(client):
    """
    Load core variables for size, leverage, profitability, R&D, and cost structure.

    Note: column 'at' is renamed to 'total_assets' to avoid conflict
    with BigQuery keyword AT.
    """

    query = f"""
    SELECT
      gvkey,
      fyear,
      sale,
      `at` AS total_assets,
      dltt,
      dlc,
      emp,
      oibdp,
      xrd,
      cogs,
      xsga,
      ppent,
      capx,
      ni
    FROM {FULL_TABLE_NAME}
    WHERE sale IS NOT NULL
      AND `at` IS NOT NULL
      AND emp IS NOT NULL
      AND oibdp IS NOT NULL
      AND xrd IS NOT NULL
      AND cogs IS NOT NULL
      AND xsga IS NOT NULL
      AND ppent IS NOT NULL
      AND capx IS NOT NULL
      AND ni IS NOT NULL
      AND fyear BETWEEN 2000 AND 2024
    """
    # If you ever want to limit number of rows for speed:
    # query += " LIMIT 20000"

    df = client.query(query).to_dataframe()
    print(f"Loaded {len(df):,} rows from BigQuery.")
    return df


# -------------------------
# FEATURE ENGINEERING
# -------------------------
def engineer_features(df):
    """
    Create log-transformed variables, leverage, and a richer set of ratios
    for use in the OLS / Ridge comparison.
    """

    df = df.copy()

    # require domain expertise (finance, acccounting), in additional to data expertise 
    # Basic log size measures
    df["log_sale"] = np.log(df["sale"])
    df["log_at"] = np.log(df["total_assets"])
    df["log_emp"] = np.log(df["emp"])
    df["log_capx"] = np.log(df["capx"])

    # Book leverage
    df["leverage"] = (df["dltt"] + df["dlc"]) / df["total_assets"]

    # Profitability and intensity measures
    df["profit_margin"] = df["oibdp"] / df["sale"]          # operating margin
    df["rd_intensity"] = df["xrd"] / df["sale"]             # R&D / sales
    df["ni_margin"] = df["ni"] / df["sale"]                 # net income / sales

    # Cost structure ratios
    df["cogs_ratio"] = df["cogs"] / df["sale"]              # COGS / sales
    df["sgna_ratio"] = df["xsga"] / df["sale"]              # SG&A / sales

    # Capital intensity
    # df["capital_intensity"] = df["ppent"] / df["total_assets"]

    # Employment per million dollars of assets
    df["emp_per_million_assets"] = df["emp"] / (df["total_assets"] / 1_000_000)

    # Clean up infinities and missing values
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # Save engineered features for inspection 
    df.to_csv("compustat_ml_features.csv", index=False)
    print("Saved engineered features to compustat_ml_features.csv")

    return df


# ======================================================
# EXERCISE 1: CORRELATION VIA MULTIVARIABLE REGRESSION
# ======================================================
def correlation_regression_exercise(df):
    """
    # Use multivariable regression to study correlation between
    firm features and log sales.

    # Standardize X and y
    # Run OLS (statsmodels)
    # Run Ridge regression (sklearn)
    # Compare coefficients
    # Save comparison as CSV and bar plot
    """

    print("\n=== Exercise 1: Correlation – Multivariable Regression ===")

    # Richer feature set to highlight Ridge vs OLS differences
    # might want to drop profit margin and sgna ratio for visualization later 
    feature_cols = [
        "log_at",
        "log_emp",
        "log_capx",
        "leverage",
        "profit_margin",
        "rd_intensity",
        "ni_margin",
        "cogs_ratio",
        "sgna_ratio",
        "emp_per_million_assets",
    ]
    target_col = "log_sale"

    X = df[feature_cols]
    y = df[target_col]

    # STANDARDIZE (so coefficients are comparable in magnitude)
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_std = scaler_X.fit_transform(X)
    y_std = scaler_y.fit_transform(y.values.reshape(-1, 1)).ravel()

    X_df = pd.DataFrame(X_std, columns=feature_cols)

    # -------------------------
    # OLS REGRESSION
    # -------------------------
    X_const = sm.add_constant(X_df)
    ols = sm.OLS(y_std, X_const).fit()

    print("\nOLS Summary:")
    print(ols.summary())

    ols_coefs = ols.params.drop("const")

    # -------------------------
    # RIDGE REGRESSION
    # -------------------------
    alpha_value = 50.0  # change this to demonstrate different levels of shrinkage; when alpha = 0, ridge = OLS 
    ridge = Ridge(alpha=alpha_value)
    ridge.fit(X_std, y_std)

    # Ridge "summary" metrics
    y_pred_ridge = ridge.predict(X_std)
    r2_ridge = r2_score(y_std, y_pred_ridge)
    mse_ridge = mean_squared_error(y_std, y_pred_ridge)

    print("\nRidge Regression Summary (sklearn)")
    print("===================================")
    print(f"Alpha (penalty):        {alpha_value}")
    print(f"Number of observations: {X_std.shape[0]}")
    print(f"Number of features:     {X_std.shape[1]}")
    print(f"R-squared:              {r2_ridge:0.4f}")
    print(f"MSE:                    {mse_ridge:0.4f}")
    print(f"Intercept:              {ridge.intercept_:0.4f}")
    print("\nRidge Coefficients (standardized):")
    for name, coef in zip(feature_cols, ridge.coef_):
        print(f"  {name:22s} {coef: .4f}")

    ridge_coefs = pd.Series(ridge.coef_, index=feature_cols)

    # -------------------------
    # COMPARE COEFFICIENTS
    # -------------------------
    coef_compare = pd.DataFrame({"OLS": ols_coefs, "Ridge": ridge_coefs})
    print("\nCoefficient Comparison (standardized):")
    print(coef_compare)

    # Save coefficient comparison to CSV
    coef_compare.to_csv("ols_vs_ridge_coefficients.csv", index=True)
    print("Saved: ols_vs_ridge_coefficients.csv")

    # -------------------------
    # PLOT COEFFICIENTS
    # -------------------------
    x = np.arange(len(feature_cols))
    width = 0.35

    plt.figure(figsize=(10, 5))
    plt.bar(x - width / 2, coef_compare["OLS"], width, label="OLS")
    plt.bar(x + width / 2, coef_compare["Ridge"], width, label="Ridge")
    plt.xticks(x, feature_cols, rotation=45, ha="right")
    plt.ylabel("Standardized Coefficient")
    plt.title(f"Exercise – Standardized Coefficients: OLS vs Ridge (alpha={alpha_value})")
    plt.legend()
    plt.tight_layout()
    plt.savefig("ols_vs_ridge_coefficients.png")
    plt.close()
    print("Saved: ols_vs_ridge_coefficients.png")

    return coef_compare


# -------------------------
# MAIN
# -------------------------
def main():
    # 1. Connect to BigQuery
    client = get_bq_client()

    # 2. Load raw Compustat data
    df_raw = load_compustat_data(client)

    # 3. Engineer features
    df_features = engineer_features(df_raw)

    # 4. Run Exercise 1 (correlation via multivariable regression)
    _ = correlation_regression_exercise(df_features)

    print("\nAll outputs saved in current directory.")


if __name__ == "__main__":
    main()
