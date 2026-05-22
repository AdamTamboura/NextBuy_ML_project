import pandas as pd

#Features Selectionnées

FEATURES = [
    "user_product_count",
    "user_recency",
    "product_popularity",
    "order_dow",
    "order_hour_of_day",
    "days_since_prior_order",
    "add_to_cart_order",
    "aisle_id",
    "department_id",
]

#Cible à prédire

TARGET = "reordered"


#Fonction original de split enlève notamment  certain data-leak du train-set.

def temporal_user_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values(["user_id", "order_number"])

    max_order_by_user = df.groupby("user_id")["order_number"].transform("max")

    train_df = df[df["order_number"] < max_order_by_user].copy()
    test_df = df[df["order_number"] == max_order_by_user].copy()

    return train_df, test_df

#fonction qui ajoute notre data engienering

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["user_product_count"] = (
        df.groupby(["user_id", "product_id"])["order_id"]
        .transform("nunique")
    )

    df["user_recency"] = (
        df.groupby("user_id")["days_since_prior_order"]
        .transform("mean")
    )

    df["product_popularity"] = (
        df.groupby("product_id")["order_id"]
        .transform("nunique")
    )

    return df

#fonction qui ajoute les features du train esentielles pour une bonne prédiction tout en évitant le data-leak

def add_test_features_from_train(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    test_df = test_df.copy()

    user_product_count = (
        train_df.groupby(["user_id", "product_id"])["order_id"]
        .nunique()
        .reset_index(name="user_product_count")
    )

    user_recency = (
        train_df.groupby("user_id")["days_since_prior_order"]
        .mean()
        .reset_index(name="user_recency")
    )

    product_popularity = (
        train_df.groupby("product_id")["order_id"]
        .nunique()
        .reset_index(name="product_popularity")
    )

    test_df = test_df.merge(
        user_product_count,
        on=["user_id", "product_id"],
        how="left",
    )

    test_df = test_df.merge(
        user_recency,
        on="user_id",
        how="left",
    )

    test_df = test_df.merge(
        product_popularity,
        on="product_id",
        how="left",
    )

    test_df["user_product_count"] = test_df["user_product_count"].fillna(0)
    test_df["user_recency"] = test_df["user_recency"].fillna(
        train_df["days_since_prior_order"].mean()
    )
    test_df["product_popularity"] = test_df["product_popularity"].fillna(0)

    return test_df

#fonction qui fait un data cleaning de base et récupérer X(paramètres) et y(cible)
def prepare_xy(df: pd.DataFrame):
    df_ml = df[FEATURES + [TARGET]].copy()
    df_ml = df_ml.dropna()
    df_ml[TARGET] = df_ml[TARGET].astype(int)

    X = df_ml[FEATURES]
    y = df_ml[TARGET]

    return X, y