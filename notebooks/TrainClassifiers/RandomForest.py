import pandas as pd
import logging
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from pathlib import Path
import joblib

#Loading train and test datasets
train_df = pd.read_csv(Path(__file__).parent.parent.parent / 'data' / 'features_train.csv')
test_df = pd.read_csv(Path(__file__).parent.parent.parent / 'data' / 'features_test.csv')

#Combine both of them
df = pd.concat([train_df, test_df], ignore_index=True)

#Separate feature and labels
x = df.drop(columns=['label'])
y = df['label']

#Handle categorical data (need to conver string->float)
categorical_cols = x.select_dtypes(include=['object']).columns
numeric_cols = x.select_dtypes(include=['int64', 'float64']).columns

if len(categorical_cols) > 0:
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_array = encoder.fit_transform(x[categorical_cols])
    encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(categorical_cols))
    x = pd.concat([x[numeric_cols].reset_index(drop=True),
                   encoded_df.reset_index(drop=True)], axis=1)
    

#Scale features
scaler = StandardScaler()
X_Scaled = scaler.fit_transform(x)

#Split data into train/Test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_Scaled, y, test_size=0.2, random_state=42, stratify=y
)

#Train Random Forest
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight='balanced',
    random_state=42,
    n_jobs=1
)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

#Logging setip
log_folder = Path(__file__).parent.parent.parent / 'loghub'
log_folder.mkdir(parents=True, exist_ok=True)

base_name = 'Rfmodel_run_rf'
extension = '.txt'
i = 1
while(log_folder / f"{base_name}_{i}{extension}").exists():
    i += 1

log_filename = log_folder / f"{base_name}_{i}{extension}"

logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 9️⃣ Evaluate and log performance
logging.info("=== Random Forest Evaluation ===")
logging.info(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
logging.info(f"Precision: {precision_score(y_test, y_pred):.4f}")
logging.info(f"Recall: {recall_score(y_test, y_pred):.4f}")
logging.info(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
logging.info(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}\n")

print(f"Evaluation logged to {log_filename}")

models_folder = Path(__file__).parent.parent.parent / 'models'
models_folder.mkdir(parents=True, exist_ok=True)
joblib.dump(rf, models_folder / 'RandomForest.pkl')