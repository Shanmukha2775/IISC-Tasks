import pandas as pd
import numpy as np

# Load the master mel spectrogram
master = pd.read_csv('mel_spectrogram.csv', header=None).values.astype(float)

# List of all submatrix files (original + 4 variants)
submatrix_files = [
    'submatrix_2.csv',
    'submatrix_2_variant_1.csv',
    'submatrix_2_variant_2.csv',
    'submatrix_2_variant_3.csv',
    'submatrix_2_variant_4.csv'
]

# Process each submatrix
for filename in submatrix_files:
    # Load the current submatrix
    sub = pd.read_csv(filename, header=None).values.astype(float)
    
    rows, cols = sub.shape  # Expected: 128 rows, 30 columns
    
    best_mse = float('inf')
    best_start_col = -1
    
    # Sliding window over columns (rows are fixed 0-127)
    for start_col in range(master.shape[1] - cols + 1):
        # Extract the corresponding slice from master
        master_slice = master[0:rows, start_col:start_col + cols]
        
        # Calculate Mean Squared Error
        mse = np.mean((master_slice - sub) ** 2)
        
        # Update if this is the best match so far
        if mse < best_mse:
            best_mse = mse
            best_start_col = start_col
    
    # Calculate end column
    end_col = best_start_col + cols - 1
    
    # Print the result in the required format
    print(f"{filename} best match:")
    print(f"row_start_row_end: 0_127")
    print(f"col_start_col_end: {best_start_col}_{end_col}")
    print(f"MSE: {best_mse:.10f}")
    print("-" * 60)