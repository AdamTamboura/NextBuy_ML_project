import pandas as pd
from pathlib import Path
import sys


#track down our project_paths.py file
ROOT = next(
    (p for p in [Path.cwd(), *Path.cwd().parents] if (p / "project_paths.py").exists()),
    None
)
if ROOT is None:
    raise RuntimeError("Racine du projet introuvable (project_paths.py non trouvé).")

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from project_paths import raw_data_file, PROCESSED_DATA_DIR # noqa: E402 because need to find the path and add to sys frist

#Load all the datasets

orders = pd.read_csv(raw_data_file("orders.csv"))
order_products = pd.read_csv(raw_data_file("order_products.csv"))
products = pd.read_csv(raw_data_file("products.csv"))
aisles = pd.read_csv(raw_data_file("aisles.csv"))
departments = pd.read_csv(raw_data_file("departments.csv"))

#Merge the dataset with

df_nextBuy = order_products.merge(orders, on="order_id", how = "left")
df_nextBuy = df_nextBuy.merge(products, on = "product_id", how = "left")
df_nextBuy = df_nextBuy.merge(aisles, on = "aisle_id", how = "left")
df_nextBuy = df_nextBuy.merge(departments, on = "department_id", how = "left")

#save the dataframe compressed in to data/processed
output_path = PROCESSED_DATA_DIR / "nextbuy.pkl.gz"
df_nextBuy.to_pickle(output_path, compression="gzip")






