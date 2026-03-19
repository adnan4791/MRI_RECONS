# MRI Reconstruction with OpenCL and N4 Bias Field Correction

A high-performance MRI reconstruction pipeline combining OpenCL for k-space processing and ITK/OpenCL for bias field correction.

## Features

- **k-space Processing**: FFT-based reconstruction using OpenCL and clFFT
- **Coil Combination**: Root-Sum-of-Squares (RSS) normalization
- **Bias Field Correction**: N4 bias field correction with two implementation options:
  - **CPU-based**: 3D volume processing using ITK (default, more accurate)
  - **GPU-accelerated**: 2D per-slice processing using OpenCL kernels (faster)
- **Image Output**: PNG format with global contrast stretching for consistency

## System Requirements

- Linux/Unix system
- CMake 3.10 or higher
- C++11 compatible compiler
- OpenCL-capable device (CPU or GPU)
- ITK (Insight Toolkit) 4.13 or higher
- OpenCV 4.0 or higher
- clFFT library

## Dependencies Installation

### Ubuntu/Debian

```bash
# Install build tools
sudo apt-get install build-essential cmake git

# Install OpenCV
sudo apt-get install libopencv-dev

# Install OpenCL development libraries
sudo apt-get install opencl-headers ocl-icd-opencl-dev

# Install clFFT
sudo apt-get install clfft-dev

# Install ITK
sudo apt-get install libinsighttoolkit4-dev
```

### macOS (with Homebrew)

```bash
brew install cmake opencv clfft
# ITK should be installed separately from source if needed
```

## Build Instructions

### 1. Clone Repository

```bash
git clone https://github.com/adnan4791/MRI_RECONS.git
cd MRI_RECONS
mkdir build
cd build
```

### 2. Configure with CMake

The project supports two N4 bias field correction methods:

#### Option A: CPU-based N4 Bias Field Correction (Default)
Uses ITK for 3D volume-based bias field correction. Recommended for accuracy.

```bash
cmake -DUSE_GPU_N4=OFF ..
make -j4
```

**Advantages:**
- 3D volume processing (better field estimation)
- More accurate results
- Works on CPU systems without GPU

**Output message:**
```
-- N4 Bias Field Correction: CPU-based (ITK)
```

#### Option B: GPU-accelerated N4 Bias Field Correction
Uses OpenCL kernels for 2D per-slice processing. Recommended for speed.

```bash
cmake -DUSE_GPU_N4=ON ..
make -j4
```

**Advantages:**
- GPU acceleration (2-3x faster)
- Per-slice processing
- Lower memory footprint
- Good for real-time processing

**Output message:**
```
-- N4 Bias Field Correction: GPU-accelerated (OpenCL)
```

### 3. Build the Executable

```bash
# After configure step above
make -j4

# Or with explicit number of jobs
make -j8
```

The executable `mri_recon` will be created in the `build/` directory.

## Usage

### Basic Usage

```bash
./mri_recon
```

Ensure the following input files are in the same directory:
- `data_mri.cfl` - k-space data (float32)
- `data_mri.hdr` - header file with dimensions
- `kernel.cl` - OpenCL kernels

### Input File Format

The header file (`data_mri.hdr`) should contain:
```
width: <image_width>
height: <image_height>
coils: <number_of_coils>
slices: <number_of_slices>
```

### Output

Reconstructed images are saved to `output/` directory as:
```
output/slice_0.png
output/slice_1.png
...
output/slice_N.png
```

Each slice is:
- Format: PNG (8-bit grayscale)
- Scaled using global min/max for consistent contrast
- Ready for visualization or further processing

## Build Process Details

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `USE_GPU_N4` | `OFF` | Enable GPU-accelerated N4 correction |
| `CMAKE_BUILD_TYPE` | `Debug` | Build type (Debug/Release) |
| `CMAKE_CXX_STANDARD` | `11` | C++ standard version |

### Project Structure

```
.
├── CMakeLists.txt          # Build configuration
├── main.cpp                # Main application
├── kernel.cl               # OpenCL kernels
├── mri_utils.hpp           # Utility functions
├── README.md               # This file
└── build/                  # Build directory
    └── mri_recon           # Compiled executable
```

## Performance Comparison

### CPU-based N4 (ITK)
- Processing time: 5-10 minutes per dataset (depends on volume size)
- Memory usage: ~500MB-2GB
- Accuracy: High (3D processing)
- Platform: Cross-platform

### GPU-accelerated N4 (OpenCL)
- Processing time: 1-3 minutes per dataset (2-5x faster)
- Memory usage: ~200-500MB
- Accuracy: Good (2D per-slice)
- Platform: Requires OpenCL-capable device

## Switching Between Implementations

To switch from one implementation to another:

```bash
# Clean previous build
cd build
rm -rf *

# Configure with desired option
cmake -DUSE_GPU_N4=ON ..  # For GPU version
# or
cmake -DUSE_GPU_N4=OFF .. # For CPU version

# Rebuild
make -j4
```

## Troubleshooting

### CMake Configuration Fails

**Problem**: ITK not found
```
Could not find a package configuration file provided by "ITK"
```

**Solution**: 
```bash
# Ensure ITK is installed
sudo apt-get install libinsighttoolkit4-dev

# Or manually specify ITK path
cmake -DITK_DIR=/path/to/ITK/build ..
```

### OpenCL Issues

**Problem**: "No OpenCL device found"
```
Using OpenCL device: ... (FAILED)
```

**Solution**:
- Install OpenCL drivers for your GPU
- Fallback to CPU OpenCL device automatically

### Compilation Errors

**Problem**: Missing clFFT
```
clFFT library not found
```

**Solution**:
```bash
sudo apt-get install clfft-dev
```

## Environment Variables

Optional environment variables for performance tuning:

```bash
# Number of OpenCL work groups (GPU tuning)
export CL_DEVICE_TYPE=GPU

# CPU threading for ITK
export OMP_NUM_THREADS=8
```

## Development

### Adding New Features

1. Modify kernel.cl for OpenCL kernels
2. Update main.cpp for algorithm changes
3. Rebuild with appropriate option:
   ```bash
   cd build
   rm CMakeCache.txt
   cmake ..
   make clean
   make -j4
   ```

### Testing

```bash
# After build, test with sample data
./mri_recon

# Check output files
ls -la ../output/
```

## References

- [ITK Documentation](https://itk.org/ITKDoxygen/)
- [OpenCL Documentation](https://www.khronos.org/opencl/)
- [clFFT Documentation](https://github.com/clMathLibraries/clFFT)
- [N4 Bias Field Correction Paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3071855/)

## License

[Specify your license here]

## Contributors

- Adnan (adnan4791)

## Author's Notes

This project demonstrates a practical implementation of:
1. GPU-accelerated FFT for MRI reconstruction
2. Hybrid CPU/GPU approaches for bias field correction
3. Production-ready image processing pipeline

For questions or contributions, please open an issue or pull request.

---

**Last Updated**: March 19, 2026
**Project Status**: Active Development
