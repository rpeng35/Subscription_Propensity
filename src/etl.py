import pandas as pd
import os 

# Define a checkpoint
CACHE_FILE = 'payments.parquet'
SOURCE_FILE = 'payments.xlsx'

#check if we can use the cached data 
def load_data(force_reload=False):
    use_cache = False
    if os.path.exists(CACHE_FILE) and not force_reload:
        source_time = os.path.getmtime(SOURCE_FILE)
        cache_time = os.path.getmtime(CACHE_FILE)
        if source_time < cache_time:
            use_cache = True
    if use_cache:
        return pd.read_parquet(CACHE_FILE)
    else:
        payments = pd.read_excel(SOURCE_FILE, dtype={"merchant": str})
        payments.to_parquet(CACHE_FILE, index=False)
    return payments 

if __name__ == "__main__":
    payments = load_data()
