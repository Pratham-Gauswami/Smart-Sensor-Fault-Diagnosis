import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Let's assume X is your feature DataFrame
# For example:
# X = df.drop(columns=['target'])

# 1️⃣ Separate categorical and numeric columns
categorical_cols = X.select_dtypes(include=['object']).columns
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns

# 2️⃣ One-hot encode categorical columns
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
encoded_array = encoder.fit_transform(X[categorical_cols])
encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(categorical_cols))

# 3️⃣ Combine numeric and encoded categorical columns
X_combined = pd.concat([X[numeric_cols].reset_index(drop=True),
                        encoded_df.reset_index(drop=True)], axis=1)

# 4️⃣ Scale everything
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_combined)

# 5️⃣ Convert back to a DataFrame (optional, but cleaner for debugging)
X_processed = pd.DataFrame(X_scaled, columns=X_combined.columns)

print("✅ Data preprocessing complete!")
print(f"Original shape: {X.shape}, Processed shape: {X_processed.shape}")
X_processed.to_csv('../data/processed_features.csv', index=False)

