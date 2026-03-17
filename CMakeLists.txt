cmake_minimum_required(VERSION 3.10)
project(MRI_Reconstruction_OrangePi5)

# Use Debug build by default for easier diagnostics
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

# 1. Atur Standar C++
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set OpenCV directory (adjust path if OpenCV is installed elsewhere)
set(OpenCV_DIR "C:/opencv/build")

# 2. Cari Package OpenCV
find_package(OpenCV REQUIRED)

# 3. Cari Package OpenCL
find_package(OpenCL REQUIRED)

# 4. Tentukan lokasi clFFT (Jika instalasi standar di /usr/local)
# Jika Anda menginstal di lokasi berbeda, sesuaikan path berikut:
include_directories(/usr/local/cuda/include)
include_directories(/usr/local/include)

# 5. Tambahkan Executable
add_executable(mri_recon main.cpp)

# 6. Hubungkan Include Directories
target_include_directories(mri_recon PRIVATE 
    ${OpenCV_INCLUDE_DIRS} 
    ${OpenCL_INCLUDE_DIRS}
)

# 7. Hubungkan Libraries
# Kita tambahkan clfft secara manual karena biasanya tidak memiliki find_package standar
target_link_libraries(mri_recon 
    ${OpenCV_LIBS} 
    ${OpenCL_LIBRARIES} 
    clFFT
)

# Pesan konfirmasi saat konfigurasi
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "OpenCL library status:")
message(STATUS "    libraries: ${OpenCL_LIBRARIES}")