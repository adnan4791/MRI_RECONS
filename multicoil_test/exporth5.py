import h5py
import numpy as np
from bart import bart

def convert_h5_to_cfl(h5_file, output_prefix, h5_key='kspace'):
    # 1. Read multicoil h5 file
    with h5py.File(h5_file, 'r') as hf:
        # Assume kspace is (coil, z, y, x) or (z, coil, y, x)
        # Adjust mapping based on your specific h5 structure
        data = np.array(hf[h5_key])
        print(f"Original shape: {data.shape}")

    # 2. Reorder for BART (kx, ky, kz, coil, 1, ..., time)
    # Often multicoil data needs transpose: (chan, z, y, x) -> (x, y, z, chan)
    # Example transposition:
    data = np.transpose(data, (3, 2, 1, 0)) 
    print(f"Transposed shape: {data.shape}")

    # 3. Save as cfl
    # Requires bart python wrapper or manual cfl writing
    # Manual write example:
    import os
    from bart.cfl import writecfl

    writecfl(output_prefix, data)
    print(f"Saved to {output_prefix}.cfl and {output_prefix}.hdr")

# Usage
# convert_h5_to_cfl('rawdata.h5', 'reconstructed_data')
