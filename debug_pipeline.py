import sys
import pandas as pd
import numpy as np
from src.utils.main_utils import load_object

# Load the artifacts from the most recent training
ohe_obj = load_object('artifact/01_26_2026_15_24_33/data_ingestion/ohe_object.pkl')
scaler_obj = load_object('artifact/01_26_2026_15_24_33/data_ingestion/scaling_object.pkl')

# Create test data
test_data = pd.DataFrame({
    'age': [35],
    'job': ['technician'],
    'marital': ['single'],
    'education': ['secondary'],
    'default': ['no'],
    'balance': [1200],
    'housing': ['yes'],
    'loan': ['no'],
    'contact': ['cellular'],
    'day': [15],
    'month': ['jan'],
    'duration': [600],
    'campaign': [2],
    'pdays': [-1],
    'previous': [0],
    'poutcome': ['unknown']
})

# Add pdays_unknown
test_data['pdays_unknown'] = test_data['pdays'].apply(lambda x: 0 if x == -1 else x)

print("Input data columns:", list(test_data.columns))
print("Input data shape:", test_data.shape)
print()

# Apply OHE
print("Applying OHE...")
ohe_result = ohe_obj.transform(test_data)
print(f"OHE output type: {type(ohe_result)}")
print(f"OHE output shape: {ohe_result.shape}")

# Get column names
ct = ohe_obj.named_steps['column transformation']
feature_names = list(ct.get_feature_names_out())
print(f"Feature names from get_feature_names_out(): {len(feature_names)} columns")
print(f"First 10 names: {feature_names[:10]}")
print()

# Create DataFrame
ohe_df = pd.DataFrame(ohe_result, columns=feature_names)
print(f"OHE DataFrame shape: {ohe_df.shape}")
print(f"OHE DataFrame columns: {list(ohe_df.columns)}")
print(f"OHE DataFrame columns count: {len(ohe_df.columns)}")
print()

# Drop columns
drop_cols = ['contact', 'day', 'month', 'pdays']
cols_to_drop = [col for col in drop_cols if col in ohe_df.columns]
print(f"Columns to drop: {cols_to_drop}")
ohe_df_dropped = ohe_df.drop(columns=cols_to_drop)
print(f"After drop - shape: {ohe_df_dropped.shape}")
print(f"After drop - columns: {list(ohe_df_dropped.columns)}")
print(f"After drop - columns count: {len(ohe_df_dropped.columns)}")
print()

# Try to apply scaler
print("Applying scaler...")
try:
    scaled = scaler_obj.transform(ohe_df_dropped)
    print(f"Scaling successful! Shape: {scaled.shape}")
except Exception as e:
    print(f"Scaling failed: {e}")
