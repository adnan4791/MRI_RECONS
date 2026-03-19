__kernel void fermi_filter(__global float2* data, int width, int height, int coils, float k_radius, float k_width) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) return;

    int pixels = width * height;
    for (int c = 0; c < coils; c++) {
        int idx = c * pixels + y * width + x;
        // Calculate distance from center (DC component in k-space)
        float dx = x - width / 2.0f;
        float dy = y - height / 2.0f;
        float r = sqrt(dx*dx + dy*dy);
        // Fermi filter: high frequencies get attenuated
        float weight = 1.0f / (1.0f + exp((r - k_radius) / k_width));
        
        data[idx] *= weight;
    }
}

__kernel void fft_shift_2d(__global float2* input, __global float2* output, int width, int height, int coils) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) return;

    int pixels = width * height;
    for (int c = 0; c < coils; c++) {
        int idx = c * pixels + y * width + x;
        
        // Calculate the source position for FFT shift
        int src_x = (x + width/2) % width;
        int src_y = (y + height/2) % height;
        int src_idx = c * pixels + src_y * width + src_x;
        
        // Copy from shifted position to current position
        output[idx] = input[src_idx];
    }
}

__kernel void rss_normalized(__global float2* input, __global float* output, int pixels, int coils, float norm) {
    int i = get_global_id(0);
    if (i >= pixels) return;

    float sum_sq = 0.0f;
    for (int c = 0; c < coils; c++) {
        float2 val = input[c * pixels + i] * norm;
        sum_sq += (val.x * val.x) + (val.y * val.y);
    }
    output[i] = sqrt(sum_sq);
}

// Kernel untuk estimasi bias field (Simple Low-pass approach)
__kernel void estimate_bias_kernel(
    __global const float* inputImage,
    __global float* biasField,
    __global float* outputImage,
    const int width,
    const int height,
    const int radius) 
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= width || y >= height) return;

    float sumLog = 0.0f;
    int count = 0;

    // Local Averaging in Log Domain (Approximating B-Spline smooth)
    for (int ky = -radius; ky <= radius; ky++) {
        for (int kx = -radius; kx <= radius; kx++) {
            int nx = x + kx;
            int ny = y + ky;

            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                float val = inputImage[ny * width + nx];
                // Tambahkan epsilon untuk menghindari log(0)
                sumLog += log(val + 1e-6f);
                count++;
            }
        }
    }

    float logBias = sumLog / count;
    biasField[y * width + x] = logBias;

    // Koreksi: I_out = I_in / exp(logBias)
    outputImage[y * width + x] = inputImage[y * width + x] / exp(logBias);
}

// ============== N4 ITK OpenCL Implementation ==============

// Gaussian kernel for smooth bias field estimation
__kernel void gaussian_blur_3x3(
    __global const float* input,
    __global float* output,
    const int width,
    const int height)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= width || y >= height) return;

    // Gaussian kernel coefficients (3x3) - simplified for compatibility
    float kernel_values[] = {1.0f/16.0f, 2.0f/16.0f, 1.0f/16.0f,
                             2.0f/16.0f, 4.0f/16.0f, 2.0f/16.0f,
                             1.0f/16.0f, 2.0f/16.0f, 1.0f/16.0f};

    float sum = 0.0f;
    int k_idx = 0;

    for (int ky = -1; ky <= 1; ky++) {
        for (int kx = -1; kx <= 1; kx++) {
            int nx = x + kx;
            int ny = y + ky;

            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                sum += input[ny * width + nx] * kernel_values[k_idx];
            }
            k_idx++;
        }
    }

    output[y * width + x] = sum;
}

// Estimate log bias field using iterative smoothing (N4-like approach)
__kernel void estimate_log_bias_field(
    __global const float* logImage,
    __global float* logBiasField,
    const int width,
    const int height,
    const int radius,
    const float convergence)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int ky, kx, nx, ny;
    float dx, dy, r_sq, sigma_sq, weight;

    if (x >= width || y >= height) return;

    // Local averaging in log domain with Gaussian weighting
    float sumWeightedLog = 0.0f;
    float sumWeights = 0.0f;

    for (ky = -radius; ky <= radius; ky++) {
        for (kx = -radius; kx <= radius; kx++) {
            nx = x + kx;
            ny = y + ky;

            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                // Gaussian weight
                dx = (float)kx;
                dy = (float)ky;
                r_sq = dx*dx + dy*dy;
                sigma_sq = (radius/2.0f) * (radius/2.0f);
                weight = exp(-r_sq / (2.0f * sigma_sq));

                sumWeightedLog += logImage[ny * width + nx] * weight;
                sumWeights += weight;
            }
        }
    }

    if (sumWeights > 0.0f) {
        logBiasField[y * width + x] = sumWeightedLog / sumWeights;
    } else {
        logBiasField[y * width + x] = logImage[y * width + x];
    }
}

// Apply bias field correction multiplicatively
__kernel void apply_bias_correction(
    __global const float* inputImage,
    __global const float* logBiasField,
    __global float* outputImage,
    const int width,
    const int height)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    float biasField = exp(logBiasField[idx]);

    // Avoid division by zero
    if (biasField < 1e-6f) {
        biasField = 1e-6f;
    }

    outputImage[idx] = inputImage[idx] / biasField;
}

// Multi-scale iterative bias field refinement (N4-like)
__kernel void iterative_bias_estimation(
    __global const float* inputImage,
    __global float* biasField,
    __global float* tempField,
    const int width,
    const int height,
    const int scale,
    const int iteration)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= width || y >= height) return;

    // Calculate image at current iteration
    int idx = y * width + x;
    
    // Downsample factor based on scale
    int radius = (1 << scale); // 2^scale
    
    // Local average in log domain
    float sumLog = 0.0f;
    int count = 0;

    for (int ky = -radius; ky <= radius; ky++) {
        for (int kx = -radius; kx <= radius; kx++) {
            int nx = x + kx;
            int ny = y + ky;

            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                float val = inputImage[ny * width + nx];
                sumLog += log(val + 1e-6f);
                count++;
            }
        }
    }

    // Smooth estimate
    float newBias = sumLog / (float)count;
    
    // Exponential weighted average with previous estimate
    float alpha = 1.0f / (iteration + 2.0f);
    tempField[idx] = biasField[idx] * (1.0f - alpha) + newBias * alpha;
}

// Threshold and refine bias field estimate
__kernel void refine_bias_estimate(
    __global const float* logBiasField,
    __global float* refinedField,
    const int width,
    const int height,
    const float threshold)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    float bias = logBiasField[idx];

    // Apply soft threshold to suppress noise
    float absVal = fabs(bias);
    if (absVal < threshold) {
        refinedField[idx] = bias * 0.5f;
    } else {
        refinedField[idx] = bias;
    }
}
