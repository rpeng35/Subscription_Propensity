import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix 

# define features
features = ['checkout_tpv', 'link_tpv', 'total_tpv', 'avg_vol','std_vol', 'tenure', 'use_checkout', 'use_payment_link']
categorical_features = ['industry', 'business_size', 'country']


def train_model(input = "merch_profile.parquet"):
    df = pd.read_parquet(input)
    X = pd.get_dummies(df[features + categorical_features], columns=categorical_features, drop_first=True)
    Y = df['is_sub']

    # split data (80/20)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=77)

    # train model
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=77)
    model.fit(X_train, Y_train)

    # evaluate model
    Y_pred_prob = model.predict_proba(X_test)[:, 1]
    Y_pred = model.predict(X_test)

    auc = roc_auc_score(Y_test, Y_pred_prob)
    print(f"AUC: {auc:.3f} (Target > 0.75)")
    print("\nClassification Report:\n", classification_report(Y_test, Y_pred))

    return model, X_train.columns, df


def feature_importance(model, X):
    # feature importance
    feature_importance = pd.DataFrame({
        'feature': X,
        'importance': model.feature_importances_
    }).sort_values(by='importance', ascending=False).head(10)
    print("\nFeature Importance (Top 10):\n", feature_importance)

    fig, ax = plt.subplots(figsize=(12,6)) 
    # plot feature importance
    sns.barplot(x='importance', y='feature', data=feature_importance, palette='viridis', ax=ax)
    ax.set_title('Top 10 Feature Importance')
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    fig.tight_layout()
    fig.savefig('feature_importance.png')



def find_candidates(model, df, X, output = "target_marchants.csv"):
    # find potential candidates/merchants with high probability of subscription
    candidates = df[df['is_sub'] == 0].copy()
    X_candidates = pd.get_dummies(candidates[features + categorical_features], columns=categorical_features, drop_first=True)
    X_candidates = X_candidates.reindex(columns=X, fill_value=0)
    candidates['propensity'] = model.predict_proba(X_candidates)[:, 1]

    # select top 1000 candidates
    selected_candidates = candidates.sort_values(by='propensity', ascending=False).head(1000)
    print(selected_candidates[['merchant', 'industry', 'total_tpv', 'propensity']].head())

    # save selected candidates
    selected_candidates = selected_candidates[['merchant', 'propensity']].copy()
    selected_candidates.to_csv(output, index=False)


if __name__ == "__main__":
    model, X_cols, df = train_model()
    feature_importance(model, X_cols)
    find_candidates(model, df, X_cols)