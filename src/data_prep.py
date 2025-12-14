import pandas as pd

# Load the data -> treat merchant as string for potential format errors
merchants = pd.read_csv("merchants.csv", dtype={"merchant": str})
payments = pd.read_parquet("payments.parquet")

# Data Inspection
print(merchants.info())
print(merchants.describe())
print(payments.info())
print(payments.describe())

joined_df = pd.merge(merchants, payments, on='merchant', how='inner')
print(joined_df.head())
print(joined_df.shape)
print(f"matched rows: {len(joined_df)}")
print(f"unique merchants: {joined_df['merchant'].nunique()}")

# save as parquet
joined_df.to_parquet("joined_df.parquet", index=False)