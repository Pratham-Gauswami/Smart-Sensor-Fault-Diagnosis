# visualization_pipeline.py

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# 1️⃣ Load the extracted features
train_df = pd.read_csv('../data/features_test.csv')

# 2️⃣ Separate features and labels
X = train_df.drop(columns=['label'])
y = train_df['label']

# 3️⃣ Handle categorical (string) columns
categorical_cols = X.select_dtypes(include=['object']).columns
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns

if len(categorical_cols) > 0:
    print(f"Encoding categorical columns: {list(categorical_cols)}")
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_array = encoder.fit_transform(X[categorical_cols])
    encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(categorical_cols))
    X = pd.concat([X[numeric_cols].reset_index(drop=True),
                   encoded_df.reset_index(drop=True)], axis=1)
else:
    print("No categorical columns detected.")

# 4️⃣ Standardize the features (important before PCA/t-SNE)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5️⃣ PCA Visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("Label counts:")
print(y.value_counts())

plt.figure(figsize=(8,6))
plt.scatter(X_pca[y==0, 0], X_pca[y==0, 1], label='Normal', alpha=0.6)
plt.scatter(X_pca[y==1, 0], X_pca[y==1, 1], label='Anomaly', alpha=0.6)
plt.title('PCA Visualization of Train Features')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()

# 6️⃣ t-SNE Visualization (nonlinear relationships)
tsne = TSNE(n_components=2, random_state=42, perplexity=30, learning_rate=200)
X_tsne = tsne.fit_transform(X_scaled)

plt.figure(figsize=(8,6))
plt.scatter(X_tsne[y==0, 0], X_tsne[y==0, 1], label='Normal', alpha=0.6)
plt.scatter(X_tsne[y==1, 0], X_tsne[y==1, 1], label='Anomaly', alpha=0.6)
plt.title('t-SNE Visualization of Train Features')
plt.xlabel('Dim1')
plt.ylabel('Dim2')
plt.legend()
plt.show()



# Option 1 — Understand Training Data Alone
# You only want to see if normal vs anomaly in training data are separable.
# → Then use only features_train.csv.