#include <arrayfire.h>
#include <cmath>
#include <iostream>

#include <vector>

using namespace af;

features get_feats(array& img_color, bool console)
{
    // Convert the image from RGB to gray-scale
    array img = colorSpace(img_color, AF_GRAY, AF_RGB);
    // For visualization in ArrayFire, color images must be in the [0.0f-1.0f] interval
    img_color /= 255.f;

    features feat = fast(img);
    // features feat = harris(img);
    // features feat = susan(img);

    // features feat;
    // array desc;
    // orb(feat, desc, img, 20.f, 800);

    return feat;
}

void feat_detect_demo(const char* filepath) {

    // Get feats using OpenCL
    setBackend(Backend::AF_BACKEND_OPENCL);
    setDevice(0);
    info();
    array img_af_opencl = loadImage(filepath, true);
    features feats_opencl = get_feats(img_af_opencl, false);
    int num_opencl_feats = feats_opencl.getNumFeatures();
    float* opencl_x = feats_opencl.getX().host<float>();
    float* opencl_y = feats_opencl.getY().host<float>();

    float opencl_x_min = af::min<float>(feats_opencl.getX());
    float opencl_x_max = af::max<float>(feats_opencl.getX());
    float opencl_y_min = af::min<float>(feats_opencl.getY());
    float opencl_y_max = af::max<float>(feats_opencl.getY());

    // Get feats using CUDA
    setBackend(Backend::AF_BACKEND_CUDA);
    setDevice(0);
    info();
    array img_af_cuda = loadImage(filepath, true);
    features feats_cuda = get_feats(img_af_cuda, false);
    int num_cuda_feats = feats_cuda.getNumFeatures();
    float* cuda_x = feats_cuda.getX().host<float>();
    float* cuda_y = feats_cuda.getY().host<float>();

    float cuda_x_min = af::min<float>(feats_cuda.getX());
    float cuda_x_max = af::max<float>(feats_cuda.getX());
    float cuda_y_min = af::min<float>(feats_cuda.getY());
    float cuda_y_max = af::max<float>(feats_cuda.getY());

    // Get feats using CPU
    setBackend(Backend::AF_BACKEND_CPU);
    setDevice(0);
    info();
    array img_af_cpu = loadImage(filepath, true);
    features feats_cpu = get_feats(img_af_cpu, false);
    // std::cout << img_af_cpu.dims() << std::endl;
    std::cout << "Rows: " << img_af_cpu.dims()[0] << std::endl;
    std::cout << "Cols: " << img_af_cpu.dims()[1] << std::endl;
    int num_cpu_feats = feats_cpu.getNumFeatures();
    std::cout << "CPU features: " << num_cpu_feats << std::endl;
    std::cout << "CUDA features: " << num_cuda_feats << std::endl;
    float* cpu_x = feats_cpu.getX().host<float>();
    float* cpu_y = feats_cpu.getY().host<float>();

    float cpu_x_min = af::min<float>(feats_cpu.getX());
    float cpu_x_max = af::max<float>(feats_cpu.getX());
    float cpu_y_min = af::min<float>(feats_cpu.getY());
    float cpu_y_max = af::max<float>(feats_cpu.getY());

    std::cout << "CPU range: " << std::endl;
    std::cout << "X: " << cpu_x_min << ","
              << cpu_x_max << std::endl;
    std::cout << "Y: " << cpu_y_min << ","
              << cpu_y_max << std::endl;

    std::cout << "CUDA range: " << std::endl;
    std::cout << "X: " << cuda_x_min << ","
              << cuda_x_max << std::endl;
    std::cout << "Y: " << cuda_y_min << ","
              << cuda_y_max << std::endl;

    std::cout << "OpenCL range: " << std::endl;
    std::cout << "X: " << opencl_x_min << ","
              << opencl_x_max << std::endl;
    std::cout << "Y: " << opencl_y_min << ","
              << opencl_y_max << std::endl;

    // Draw draw_len x draw_len crosshairs where the corners are
    const int draw_len = 10;
    for (size_t f = 0; f < num_cpu_feats; f++) {
        int x = cpu_x[f];
        int y = cpu_y[f];
        img_af_cpu(x, seq(y-draw_len, y+draw_len), 0) = 0.f;
        img_af_cpu(x, seq(y-draw_len, y+draw_len), 1) = 1.f;
        img_af_cpu(x, seq(y-draw_len, y+draw_len), 2) = 0.f;
    }

    for (size_t f = 0; f < num_cuda_feats; f++) {
        // Draw vertical line of (draw_len * 2 + 1) pixels centered on  the corner
        // Set only the first channel to 1 (green lines)
        int x = cuda_x[f];
        int y = cuda_y[f];
        img_af_cpu(seq(x-draw_len, x+draw_len), y, 0) = 1.f;
        img_af_cpu(seq(x-draw_len, x+draw_len), y, 1) = 1.f;
        img_af_cpu(seq(x-draw_len, x+draw_len), y, 2) = 0.f;
    }

    Window wnd("Feature Detection Demo");
    while(!wnd.close())
        wnd.image(img_af_cpu);
}

int main(int argc, char** argv )
{
    std::cout << "*** Two Squares Horizontal ***" << std::endl;
    feat_detect_demo("squares_horiz.jpg");

    std::cout << "*** Two Squares Vertical ***" << std::endl;
    feat_detect_demo("squares_vert.jpg");

    return 0;
}
