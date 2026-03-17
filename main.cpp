#include <iostream>
#include <vector>
#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>
#include <clFFT.h>
#include <opencv2/opencv.hpp>
#include "mri_utils.hpp"

// Define OpenCL vector types if not available
typedef cl_float2 float2;

int main() {
    std::string base_path = "data_mri"; // Sesuaikan nama file Anda
    MRI_Dims dims = read_hdr(base_path);
    if (dims.width == 0) { std::cerr << "Gagal baca HDR!"; return -1; }

    // --- SETUP OPENCL ---
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.empty()) {
        std::cerr << "No OpenCL platforms found" << std::endl;
        return -1;
    }

    std::vector<cl::Device> devices;
    // Force CPU devices only since GPU is not available
    platforms[0].getDevices(CL_DEVICE_TYPE_CPU, &devices);
    cl_device_type chosenType = CL_DEVICE_TYPE_CPU;
    if (devices.empty()) {
        std::cerr << "No OpenCL CPU devices found on platform: " << platforms[0].getInfo<CL_PLATFORM_NAME>() << std::endl;
        return -1;
    }

    cl::Device device = devices[0];
    std::cout << "Using OpenCL device: " << device.getInfo<CL_DEVICE_NAME>() << " (CPU)" << std::endl;

    cl::Context context(device);
    cl::CommandQueue queue(context, device);

    // Load Kernel dari file
    std::ifstream k_file("kernel.cl");
    std::string k_src((std::istreambuf_iterator<char>(k_file)), std::istreambuf_iterator<char>());
    cl::Program program(context, k_src);
    try {
        program.build({device});
    } catch (const cl::Error &e) {
        std::cerr << "OpenCL program build failed: " << e.what() << " (" << e.err() << ")\n";
        std::string log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
        std::cerr << "Build log:\n" << log << std::endl;
        return -1;
    }

    // --- SETUP clFFT ---
    clfftSetupData fftSetup;
    clfftInitSetupData(&fftSetup);
    clfftSetup(&fftSetup);
    clfftPlanHandle plan;
    size_t cl_dims[2] = {dims.width, dims.height};
    clfftCreateDefaultPlan(&plan, context(), CLFFT_2D, cl_dims);
    clfftSetPlanBatchSize(plan, dims.coils);
    clfftBakePlan(plan, 1, &queue(), NULL, NULL);

    // --- BUFFERS ---
    size_t slice_size = dims.width * dims.height;
    cl::Buffer buf_kspace(context, CL_MEM_READ_WRITE, slice_size * dims.coils * sizeof(float2));
    cl::Buffer buf_temp(context, CL_MEM_READ_WRITE, slice_size * dims.coils * sizeof(float2));  // Temporary buffer for FFT shift
    cl::Buffer buf_out(context, CL_MEM_WRITE_ONLY, slice_size * sizeof(float));

    std::ifstream cfl(base_path + ".cfl", std::ios::binary);
    std::vector<std::complex<float>> host_data(slice_size * dims.coils);

    for (size_t s = 0; s < dims.slices; ++s) {
        cfl.read((char*)host_data.data(), host_data.size() * 8);
        queue.enqueueWriteBuffer(buf_kspace, CL_TRUE, 0, host_data.size() * 8, host_data.data());

        // 1. FFT shift k-space (move DC from center to (0,0))
        cl::Kernel k_shift1(program, "fft_shift_2d");
        k_shift1.setArg(0, buf_kspace);
        k_shift1.setArg(1, buf_temp);
        k_shift1.setArg(2, (int)dims.width);
        k_shift1.setArg(3, (int)dims.height);
        k_shift1.setArg(4, (int)dims.coils);
        queue.enqueueNDRangeKernel(k_shift1, 0, cl::NDRange(dims.width, dims.height));
        queue.finish();

        // 2. Fermi filter
        cl::Kernel k_fermi(program, "fermi_filter");
        k_fermi.setArg(0, buf_temp);
        k_fermi.setArg(1, (int)dims.width);
        k_fermi.setArg(2, (int)dims.height);
        k_fermi.setArg(3, (int)dims.coils);
        k_fermi.setArg(4, 150.0f);  // More conservative radius (93% of Nyquist)
        k_fermi.setArg(5, 30.0f);   // Smoother transition
        queue.enqueueNDRangeKernel(k_fermi, 0, cl::NDRange(dims.width, dims.height));
        queue.finish();

        // 3. Copy filtered k-space back to buf_kspace
        queue.enqueueCopyBuffer(buf_temp, buf_kspace, 0, 0, slice_size * dims.coils * sizeof(float2));
        queue.finish();

        // 4. IFFT (in-place on kspace)
        clfftEnqueueTransform(plan, CLFFT_BACKWARD, 1, &queue(), 0, NULL, NULL, &buf_kspace(), NULL, NULL);

        // 5. FFT shift image (move DC from (0,0) to center)
        cl::Kernel k_shift2(program, "fft_shift_2d");
        k_shift2.setArg(0, buf_kspace);
        k_shift2.setArg(1, buf_temp);
        k_shift2.setArg(2, (int)dims.width);
        k_shift2.setArg(3, (int)dims.height);
        k_shift2.setArg(4, (int)dims.coils);
        queue.enqueueNDRangeKernel(k_shift2, 0, cl::NDRange(dims.width, dims.height));
        queue.finish();

        // 3. RSS + Normalisasi
        cl::Kernel k_rss(program, "rss_normalized");
        k_rss.setArg(0, buf_temp); k_rss.setArg(1, buf_out);
        k_rss.setArg(2, (int)slice_size); k_rss.setArg(3, (int)dims.coils);
        k_rss.setArg(4, 1.0f/(dims.width*dims.height));
        try {
            queue.enqueueNDRangeKernel(k_rss, 0, cl::NDRange(slice_size));
            queue.finish();
        } catch (const cl::Error &e) {
            std::cerr << "Failed to enqueue rss_normalized kernel: " << e.what() << " (" << e.err() << ")\n";
            return -1;
        }

        // 4. Save dengan OpenCV
        std::vector<float> final_img(slice_size);
        queue.enqueueReadBuffer(buf_out, CL_TRUE, 0, slice_size * 4, final_img.data());
        
        cv::Mat mat(dims.height, dims.width, CV_32FC1, final_img.data());
        
        // Better scaling: use percentile-based contrast stretching
        cv::Mat sorted_mat;
        cv::sort(mat.reshape(1, 1), sorted_mat, cv::SORT_ASCENDING);
        float p1 = sorted_mat.at<float>(sorted_mat.total() * 0.01);  // 1st percentile
        float p99 = sorted_mat.at<float>(sorted_mat.total() * 0.99); // 99th percentile
        
        cv::Mat scaled;
        if (p99 <= p1) {
            // Fallback to min/max scaling
            double min_val, max_val;
            cv::minMaxLoc(mat, &min_val, &max_val);
            mat.convertTo(scaled, CV_8UC1, 255.0 / (max_val - min_val), -255.0 * min_val / (max_val - min_val));
        } else {
            mat.convertTo(scaled, CV_8UC1, 255.0 / (p99 - p1), -255.0 * p1 / (p99 - p1));
        }
        
        cv::imwrite("output/slice_" + std::to_string(s) + ".png", scaled);
        
        std::cout << "Processed slice " << s << std::endl;
    }

    clfftDestroyPlan(&plan);
    clfftTeardown();
    return 0;
}