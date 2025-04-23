import os
import pandas as pd
import numpy as np
import re
from methods.bpat import BPAT  # Adjust if needed

# --- Configuration ---
input_csv = 'sample_cell_data.csv'
marker_column = 'CD11c (Opal 520): Mean'
output_dir = 'bpat_output'
os.makedirs(output_dir, exist_ok=True)

# --- Load Data ---
df = pd.read_csv(input_csv)

if marker_column not in df.columns:
    raise ValueError(f"Marker column '{marker_column}' not found in the input file.")

marker_data = df[marker_column].dropna().to_numpy()

# --- Clean filename for plot ---
plot_basename = re.sub(r'[^a-zA-Z0-9_]', '_', marker_column)
plot_path = os.path.join(output_dir, plot_basename)

# --- Run BPAT ---
threshold = BPAT(marker_data, plot_path)
print(f"BPAT threshold for '{marker_column}': {threshold:.4f}")

# --- Add threshold column (same value repeated for all rows) ---
df[f'{marker_column}_BPAT_threshold'] = threshold

# --- Add 'positive' or 'negative' classification ---
df[f'{marker_column}_BPAT_class'] = df[marker_column].apply(
    lambda x: 'positive' if x > threshold else 'negative'
)

# --- Save updated CSV ---
output_csv = os.path.join(output_dir, 'sample_cell_data_with_BPAT.csv')
df.to_csv(output_csv, index=False)
print(f"Annotated data saved to: {output_csv}")
