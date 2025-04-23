# BPAT: Bi-phase Adaptive Thresholding for mIF Marker Intensities

**BPAT** (Bi-phase Adaptive Thresholding) is a systematic and automated thresholding approach to identify marker-positive cells from multiplexed immunofluorescence (mIF) images 

## How to Use

### 1. Install dependencies
This code uses standard scientific Python packages. Install them via pip:

```
pip install numpy pandas scikit-learn matplotlib scikit-image
```

### 2. Run the example
You can use our example cell data to test the algorithm
```
python run_bpat_example.py
```
This will:

1. Load sample_cell_data.csv
2. Apply BPAT to the column "CD11c (Opal 520): Mean"
3. Generate a thresholding plot
4. Save a new CSV with the computed threshold and cell classifications (positive / negative)

## Customisation

You can modify:
1. The input CSV path
2. The marker column name
3. The output folder

by editing the variables at the top of ```run_bpat_example.py.```