import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# ----------------------------
# STEP 1: Load All .pkl Files
# ----------------------------

data_folder = "data"
pkl_files = [f for f in os.listdir(data_folder) if f.endswith(".pkl")]

all_dataframes = []

for file in pkl_files:
    file_path = os.path.join(data_folder, file)
    df = pd.read_pickle(file_path)
    all_dataframes.append(df)

# Combine all dataframes
full_df = pd.concat(all_dataframes, ignore_index=True)
print("✅ Combined dataset shape:", full_df.shape)

# ---------------------------------------
# STEP 2: Preprocessing & Column Cleanup
# ---------------------------------------

# Drop unused columns (modify as needed)
columns_to_drop = ['TRANSACTION_ID', 'TX_DATETIME']  # Drop only if present
full_df.drop(columns=[col for col in columns_to_drop if col in full_df.columns], inplace=True)

# Drop rows with missing values
full_df.dropna(inplace=True)

# Check if target column exists
target_col = "TX_FRAUD_SCENARIO"
if target_col not in full_df.columns:
    raise ValueError(f"❌ Target column '{target_col}' not found!")

X = full_df.drop(target_col, axis=1)
y = full_df[target_col]

# --------------------------------
# STEP 3: Sample and Train-Test Split
# --------------------------------

# Take a sample for faster training
X_sample = X.sample(n=100000, random_state=42)
y_sample = y.loc[X_sample.index]

X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.2, random_state=42)

# ------------------------
# STEP 4: Model Training
# ------------------------

model = RandomForestClassifier(n_jobs=-1, random_state=42)
model.fit(X_train, y_train)

# ------------------------
# STEP 5: Evaluation
# ------------------------

y_pred = model.predict(X_test)

print("\n✅ Classification Report:\n", classification_report(y_test, y_pred))
print("✅ Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ------------------------
# STEP 6: Save Model
# ------------------------

joblib.dump(model, "model.pkl")
print("✅ Model saved to model.pkl")

# ------------------------
# STEP 7: Predict Sample
# ------------------------

sample = X_test.iloc[:1]
sample_prediction = model.predict(sample)
print("✅ Sample prediction:", sample_prediction)
