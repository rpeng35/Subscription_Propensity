import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu, chi2_contingency 


# Load data
df = pd.read_parquet("merch_profile.parquet")

# check for outliers
check_cols = ['sub_tpv', 'checkout_tpv', 'link_tpv', 'total_tpv', 'avg_vol', 'max_vol', 'min_vol', 'std_vol', 'count_vol', 'tenure']
print(df[check_cols].quantile([0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999]).round(2))

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.boxplot(x=df['total_tpv'], color='blue')
plt.title('Total TPV (log scale)')
plt.xlabel('Total TPV ($)')
plt.xscale('log')

plt.subplot(1, 2, 2)
sns.boxplot(x=df['tenure'], color='green')
plt.title('Tenure Days')

plt.tight_layout()
plt.show()

# identify extreme outliers
negative_tenure = df[df['tenure'] < 0]
df = df[df['tenure'] >= 0]
massive_whales = df[df['total_tpv'] > df['total_tpv'].quantile(0.999)]
print(f"negative tenure: {len(negative_tenure)}")
print(f"massive whales (Top 0.1%): {len(massive_whales)}")

# inspect top 10 largest merchants
print(massive_whales[['merchant', 'total_tpv', 'industry', 'tenure']].sort_values(by='total_tpv', ascending=False).head(10))

sns.set_style("whitegrid")
plt.figure(figsize=(12,5))

ind_conversion_rate = df.groupby('industry')['is_sub'].mean().sort_values(ascending=False).reset_index()
plt.subplot(1,2,1)
sns.barplot(x='is_sub', y='industry', data=ind_conversion_rate, palette='viridis')
plt.title('Industry Adoption Rate by Industry')
plt.xlabel('Adoption Rate (0.0 - 1.0)')
plt.ylabel('Industry')

# divide into size buckets
size_order = ['small', 'medium', 'large']
size_convert = df.groupby('business_size')['is_sub'].mean().reindex(size_order).reset_index()
plt.subplot(1,2,2)
sns.barplot(x='business_size', y='is_sub', data=size_convert, palette='magma')
plt.title('Adoption Rate by Business Size')
plt.xlabel('Business Size')
plt.ylabel('Adoption Rate (0.0 - 1.0)')
plt.tight_layout()
plt.show()


# boxplot separation checks
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
sns.boxplot(x='is_sub', y='total_tpv', data=df, palette='viridis', showfliers=False)
plt.yscale('log')
plt.title('Total TPV Separation by Subscription Status')

plt.subplot(1,2,2)
sns.boxplot(x='is_sub', y='tenure', data=df, palette='magma', showfliers=False)
plt.title('Tenure Separation by Subscription Status')
plt.tight_layout()
plt.show()


# Correlation matrix
check_corr_cols = ['is_sub', 'sub_tpv', 'checkout_tpv', 'link_tpv', 'total_tpv', 'avg_vol', 'std_vol', 'tenure']
corr_matrix = df[check_corr_cols].corr()

plt.figure(figsize=(12,8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlation Matrix')
plt.show()


# Mann-Whitney U test for Tenure
sub_tenure = df[df['is_sub'] == 1]['tenure']
non_sub_tenure = df[df['is_sub'] == 0]['tenure']
stat, p = mannwhitneyu(sub_tenure, non_sub_tenure, alternative='two-sided')
print(f"Tenure Effect (Mann-Whitney U): {stat}")
print(f"p-value: {p:.5f}")
print(f"median tenure (subscribers): {sub_tenure.median():.0f} days")
print(f"median tenure (non-subscribers): {non_sub_tenure.median():.0f} days")

# correlation investigation between products
overlap_checkout_link = df[(df['checkout_tpv'] > 0) & (df['link_tpv'] > 0)]
print(f"number of merchants using checkout and link: {len(overlap_checkout_link)}")
print(f"Correlation - Checkout and Link: {df['checkout_tpv'].corr(df['link_tpv']):.4f}")

overlap_checkout_sub = df[(df['checkout_tpv'] > 0) & (df['sub_tpv'] > 0)]
print(f"number of merchants using checkout and subscription: {len(overlap_checkout_sub)}")
print(f"Correlation - Checkout and Subscription: {df['checkout_tpv'].corr(df['sub_tpv']):.4f}")

overlap_link_sub = df[(df['link_tpv'] > 0) & (df['sub_tpv'] > 0)]
print(f"number of merchants using link and subscription: {len(overlap_link_sub)}")
print(f"Correlation - Link and Subscription: {df['link_tpv'].corr(df['sub_tpv']):.4f}")