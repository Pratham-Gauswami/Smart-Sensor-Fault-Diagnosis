# visualization_pipeline.py
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# 1️⃣ Load both feature CSVs
train_df = pd.read_csv('../data/features_train.csv')
test_df = pd.read_csv('../data/features_test.csv')

# Add a column so we can visualize domain differences
train_df['split'] = 'train'
test_df['split'] = 'test'

# Combine both into one DataFrame
all_df = pd.concat([train_df, test_df], ignore_index=True)


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

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4️⃣ PCA Visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8,6))
plt.scatter(X_pca[y==0, 0], X_pca[y==0, 1], label='Normal', alpha=0.6, c='blue')
plt.scatter(X_pca[y==1, 0], X_pca[y==1, 1], label='Anomaly', alpha=0.6, c='red')
plt.title('PCA Visualization: Normal vs Anomaly')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()

# 5️⃣ t-SNE Visualization (optional, takes longer)
tsne = TSNE(n_components=2, random_state=42, perplexity=30, learning_rate=200)
X_tsne = tsne.fit_transform(X_scaled)

plt.figure(figsize=(8,6))
plt.scatter(X_tsne[y==0, 0], X_tsne[y==0, 1], label='Normal', alpha=0.6, c='blue')
plt.scatter(X_tsne[y==1, 0], X_tsne[y==1, 1], label='Anomaly', alpha=0.6, c='red')
plt.title('t-SNE Visualization: Normal vs Anomaly')
plt.xlabel('Dim1')
plt.ylabel('Dim2')
plt.legend()
plt.show()

# 6️⃣ (Optional) Compare Train vs Test Distribution
plt.figure(figsize=(8,6))
plt.scatter(X_pca[splits=='train', 0], X_pca[splits=='train', 1], label='Train', alpha=0.6, c='green')
plt.scatter(X_pca[splits=='test', 0], X_pca[splits=='test', 1], label='Test', alpha=0.6, c='orange')
plt.title('PCA Visualization: Train vs Test Domain Shift')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()


# Option 2 — Compare Train vs Test Distributions
# You want to see if the feature space of train vs test (source vs target) differs — i.e., check for domain shift.
# → Then load both CSVs, merge them, and add a column indicating their source (“train” vs “test”).
# Have some faults here need to check