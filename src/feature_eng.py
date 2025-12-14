import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

def reformat_df(df):
    # Load the data
    joined_df = pd.read_parquet(df)

    # Fix Data Types
    # convert cents to dollars
    volume_cols = ['subscription_volume', 'checkout_volume', 'payment_link_volume', 'total_volume']
    for col in volume_cols:
        joined_df[col] = joined_df[col] / 100.0

    # convert date in strings to datetime objects
    joined_df['first_charge_date'] = pd.to_datetime(joined_df['first_charge_date'], errors='coerce')
    joined_df['date'] = pd.to_datetime(joined_df['date'], errors='coerce')
    print(f"invalid dates: {joined_df['first_charge_date'].isnull().sum()}")

    # drop rows with null dates
    joined_df = joined_df.dropna(subset=['first_charge_date', 'date'])

    return joined_df

def create_merchant_profile(input_df = "joined_df.parquet", output_df = "merch_profile.parquet"):
    # Load the data
    joined_df = reformat_df(input_df)

    # group by merchant
    merch_profile = joined_df.groupby('merchant').agg({
    #volume metrics
    'subscription_volume': 'sum',
    'checkout_volume': 'sum',
    'payment_link_volume': 'sum',
    'total_volume': ['sum', 'mean', 'max', 'min', 'std', 'count'],

    # date
    'date': ['min', 'max'],

    # other attributes
    'industry': 'first',
    'first_charge_date': 'first',
    'business_size': 'first',
    'country': 'first'
}).reset_index()

    merch_profile.columns = [
    'merchant',
    'sub_tpv',
    'checkout_tpv',
    'link_tpv',
    'total_tpv',
    'avg_vol',
    'max_vol',
    'min_vol',
    'std_vol',
    'count_vol',
    'date_min',
    'date_max',
    'industry',
    'sign_up_date',
    'business_size',
    'country'
]

    # is the merchant a subscription merchant?
    merch_profile['is_sub'] = (merch_profile['sub_tpv'] > 0).astype(int)

    # tenure
    max_date = joined_df['date'].max()
    merch_profile['tenure'] = (max_date - merch_profile['sign_up_date']).dt.days
    merch_profile = merch_profile[merch_profile['tenure'] >= 0]
    merch_profile = merch_profile.fillna(0)

    # volatility
    merch_profile['std_vol'] = merch_profile['std_vol'].fillna(0)

    # do they use other services?
    merch_profile['use_checkout'] = (merch_profile['checkout_tpv'] > 0).astype(int)
    merch_profile['use_payment_link'] = (merch_profile['link_tpv'] > 0).astype(int)
    merch_profile['use_both'] = (merch_profile['checkout_tpv'] > 0) & (merch_profile['link_tpv'] > 0).astype(int)

    #inspect 
    print(merch_profile.head())
    print(merch_profile.shape)   
    merch_profile.to_parquet(output_df, index=False)

    return merch_profile

def chi_square_test(merch_profile):
    # chi-square test for business size independence
    contingency_table = pd.crosstab(merch_profile['business_size'], merch_profile['is_sub'])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    print(f"Business Size Effect (Chi-square): {chi2}")
    print(f"p-value: {p:.5f}")

    # chi-square test for industry independence
    contingency_table = pd.crosstab(merch_profile['industry'], merch_profile['is_sub'])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    print(f"Industry Effect (Chi-square): {chi2}")
    print(f"p-value: {p:.5f}")

if __name__ == "__main__":
    merch_profile = create_merchant_profile()
    print(f"subscribers vs non-subscribers: {merch_profile['is_sub'].value_counts(normalize=True)}")
    chi_square_test(merch_profile)
