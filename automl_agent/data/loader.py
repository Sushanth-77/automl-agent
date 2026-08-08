"""
Dataset loaders for the three benchmark datasets.

Each loader:
  1. Tries to load from data/raw/ (already downloaded)
  2. Falls back to fetching from sklearn / seaborn / UCI
  3. Saves to data/raw/ as a CSV for future runs
  4. Returns (dataframe, target_column, dataset_description)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple

import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


class DatasetInfo(NamedTuple):
    df: pd.DataFrame
    target_column: str
    description: str
    dataset_path: str


# ── Titanic ───────────────────────────────────────────────────────────────────

def load_titanic() -> DatasetInfo:
    """
    Titanic: binary classification (survived).
    Messy: ~20% null Age, 77% null Cabin, mixed dtypes.
    Source: seaborn built-in dataset.
    """
    local_path = RAW_DIR / "titanic.csv"

    if local_path.exists():
        df = pd.read_csv(local_path)
        logger.info(f"  Loaded Titanic from local: {df.shape}")
    else:
        try:
            import seaborn as sns
            df = sns.load_dataset("titanic")
            # seaborn uses lowercase column names
            df = df.rename(columns={
                "survived": "Survived",
                "pclass": "Pclass",
                "sex": "Sex",
                "age": "Age",
                "sibsp": "SibSp",
                "parch": "Parch",
                "fare": "Fare",
                "embarked": "Embarked",
                "class": "Class",
                "who": "Who",
                "adult_male": "AdultMale",
                "deck": "Deck",
                "embark_town": "EmbarkTown",
                "alive": "Alive",
                "alone": "Alone",
            })
            # Drop seaborn-specific redundant columns
            df = df.drop(columns=["Alive", "Who", "AdultMale", "Class", "EmbarkTown", "Alone"],
                         errors="ignore")
            df.to_csv(local_path, index=False)
            logger.info(f"  Fetched Titanic from seaborn: {df.shape}")
        except Exception as e:
            logger.error(f"  Failed to load Titanic: {e}")
            raise

    return DatasetInfo(
        df=df,
        target_column="Survived",
        description=(
            "Titanic passenger survival dataset. Binary classification. "
            "~20% null Age, 77% null Deck/Cabin. Mixed numeric/categorical. "
            "891 rows, ~10 features after cleaning."
        ),
        dataset_path=str(local_path),
    )


# ── Adult Income ─────────────────────────────────────────────────────────────

def load_adult_income() -> DatasetInfo:
    """
    Adult Income (Census): binary classification (income >50K).
    Messy: '?' values treated as nulls, categorical imbalance, 14 features.
    Source: sklearn / UCI.
    """
    local_path = RAW_DIR / "adult_income.csv"

    if local_path.exists():
        df = pd.read_csv(local_path)
        logger.info(f"  Loaded Adult Income from local: {df.shape}")
    else:
        try:
            from sklearn.datasets import fetch_openml
            dataset = fetch_openml(name="adult", version=2, as_frame=True, parser="auto")
            df = dataset.frame
            # Rename target to something cleaner
            df = df.rename(columns={"class": "income"})
            # Replace '?' with NaN
            df = df.replace("?", pd.NA)
            # Normalise target: '>50K' → 1, '<=50K' → 0
            df["income"] = (df["income"].str.strip().str.replace(".", "", regex=False) == ">50K").astype(int)
            df.to_csv(local_path, index=False)
            logger.info(f"  Fetched Adult Income from sklearn: {df.shape}")
        except Exception as e:
            logger.warning(f"  sklearn fetch failed ({e}), trying pandas read_csv from UCI...")
            url = (
                "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
            )
            cols = [
                "age", "workclass", "fnlwgt", "education", "education_num",
                "marital_status", "occupation", "relationship", "race", "sex",
                "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"
            ]
            df = pd.read_csv(url, header=None, names=cols, na_values=" ?")
            df["income"] = (df["income"].str.strip() == ">50K").astype(int)
            df.to_csv(local_path, index=False)
            logger.info(f"  Fetched Adult Income from UCI: {df.shape}")

    return DatasetInfo(
        df=df,
        target_column="income",
        description=(
            "Adult Income (Census) dataset. Binary classification (income >50K). "
            "~7% nulls (workclass, occupation, native_country). "
            "Class imbalance: ~75% <=50K. 48k rows, 14 features."
        ),
        dataset_path=str(local_path),
    )


# ── House Prices (Regression) ─────────────────────────────────────────────────

def load_house_prices() -> DatasetInfo:
    """
    House Prices: regression (SalePrice).
    Messy: many features with >40% nulls, mixed types, skewed target.
    Source: sklearn / OpenML Ames Housing.
    """
    local_path = RAW_DIR / "house_prices.csv"

    if local_path.exists():
        df = pd.read_csv(local_path)
        logger.info(f"  Loaded House Prices from local: {df.shape}")
    else:
        try:
            from sklearn.datasets import fetch_openml
            dataset = fetch_openml(name="house_prices", version=1, as_frame=True, parser="auto")
            df = dataset.frame
            # Ensure target is numeric
            if "SalePrice" in df.columns:
                df["SalePrice"] = pd.to_numeric(df["SalePrice"], errors="coerce")
            df.to_csv(local_path, index=False)
            logger.info(f"  Fetched House Prices from OpenML: {df.shape}")
        except Exception as e:
            logger.warning(f"  OpenML fetch failed ({e}), trying Ames Housing CSV...")
            try:
                url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv"
                cols = [
                    "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
                    "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"
                ]
                df = pd.read_csv(url, header=None, names=cols)
                df = df.rename(columns={"MEDV": "SalePrice"})
                df.to_csv(local_path, index=False)
                logger.info(f"  Fetched Boston Housing from GitHub: {df.shape}")
            except Exception as e2:
                raise RuntimeError(
                    f"House Prices dataset could not be loaded: {e2}. "
                    "Please download manually from https://www.kaggle.com/c/house-prices-advanced-regression-techniques "
                    "and place as data/raw/house_prices.csv"
                ) from e2

    # Make sure SalePrice column exists
    target_col = "SalePrice" if "SalePrice" in df.columns else df.columns[-1]

    return DatasetInfo(
        df=df,
        target_column=target_col,
        description=(
            "House Prices dataset. Regression (SalePrice). "
            "Many features with high null rates. Skewed target. "
            "~1500 rows, 79+ features."
        ),
        dataset_path=str(local_path),
    )


# ── Registry ──────────────────────────────────────────────────────────────────

DATASET_REGISTRY: dict[str, callable] = {
    "titanic": load_titanic,
    "adult": load_adult_income,
    "adult_income": load_adult_income,
    "house_prices": load_house_prices,
    "houses": load_house_prices,
}


def load_dataset(name: str) -> DatasetInfo:
    """Load a dataset by name from the registry."""
    name = name.lower().strip()
    if name not in DATASET_REGISTRY:
        available = ", ".join(DATASET_REGISTRY.keys())
        raise ValueError(
            f"Unknown dataset '{name}'. Available: {available}"
        )
    return DATASET_REGISTRY[name]()


def load_from_path(file_path: str, target_column: str) -> DatasetInfo:
    """
    Load any CSV or Parquet file from disk as a DatasetInfo.

    Used by the Streamlit dashboard custom-upload flow.
    Returns DatasetInfo with description auto-generated from the dataframe shape.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(file_path)
    elif path.suffix.lower() in (".csv", ".tsv"):
        sep = "\t" if path.suffix.lower() == ".tsv" else ","
        df = pd.read_csv(file_path, sep=sep)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}. Use CSV or Parquet.")

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in {path.name}. "
            f"Available columns: {list(df.columns)}"
        )

    n_unique_target = df[target_column].nunique()
    task_hint = "regression" if n_unique_target > 20 else "classification"

    description = (
        f"Custom dataset: {path.name}. "
        f"{df.shape[0]} rows, {df.shape[1]} columns. "
        f"Target: '{target_column}' ({n_unique_target} unique values → likely {task_hint})."
    )
    logger.info(f"  Loaded custom dataset from {file_path}: {df.shape}")

    return DatasetInfo(
        df=df,
        target_column=target_column,
        description=description,
        dataset_path=str(path),
    )
