#ifndef MRI_UTILS_HPP
#define MRI_UTILS_HPP

#include <vector>
#include <string>
#include <fstream>
#include <complex>

struct MRI_Dims {
    size_t width, height, slices, coils;
};

// Fungsi sederhana membaca dimensi dari .hdr (Format: 320 320 256 8 1)
MRI_Dims read_hdr(const std::string& filename) {
    std::ifstream file(filename + ".hdr");
    MRI_Dims dims = {0, 0, 0, 0};
    if (file.is_open()) {
        std::string dummy; 
        std::getline(file, dummy); // Lewati komentar # Dimensions
        file >> dims.width >> dims.height >> dims.slices >> dims.coils;
    }
    return dims;
}

#endif