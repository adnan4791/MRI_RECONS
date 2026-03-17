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