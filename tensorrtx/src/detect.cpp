#include <iostream>
#include <chrono>
#include <cmath>
#include "cuda_utils.h"
#include "logging.h"
#include "common.hpp"
#include "utils.h"
#include "preprocess.h"
#include "yolov5-detect.h"

int main(int argc, char **argv)
{
    cudaSetDevice(0);

    std::string engine_name = "/home/westwell/Desktop/yolov5-5.0-tensorrt_result/engine_model/container_object.engine";
    yolov5 *det = new yolov5(engine_name, 7);

    cv::Mat img = cv::imread("/cv/work/xianghao_train/container_object_train/yolov5-5.0/data/images/0a96969233337363037dd49c023f1ac3_2017-10-25.jpg");
    int w = img.cols;
    int h = img.rows;
    unsigned char *d_image;
    cudaMalloc((void **)&d_image, sizeof(unsigned char) * w * h * 3);
    cudaMemcpy(d_image, img.data, w * h * 3 * sizeof(unsigned char),cudaMemcpyHostToDevice);

    det->detect(d_image, w, h,img);

    cudaFree(d_image);
    return 0;
}
