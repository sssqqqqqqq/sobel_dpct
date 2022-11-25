#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <cmath>

using namespace std;
using namespace cv;
 
//GPU sobel
//  x0 x1 x2
//  x3 x4 x5
//  x6 x7 x8
void sobel_gpu(unsigned char *in, unsigned char *out, int imgHeight, int imgWidth,
               sycl::nd_item<3> item_ct1)
{
    int x = item_ct1.get_local_id(2) +
            item_ct1.get_local_range(2) * item_ct1.get_group(2);
    int y = item_ct1.get_local_id(1) +
            item_ct1.get_local_range(1) * item_ct1.get_group(1);

    int index = y * imgWidth + x;
 
    int Gx = 0;
    int Gy = 0;
 
    unsigned char x0, x1, x2, x3, x4, x5, x6,x7,x8;
    if (x > 0 && x < imgWidth-1 && y > 0 && y < imgHeight-1)
    {
        x0 = in[(y-1)* imgWidth + x - 1];
        x1 = in[(y-1)* imgWidth + x ];
        x2 = in[(y-1)* imgWidth + x + 1];
 
        x3 = in[y* imgWidth + x - 1];
        x4 = in[y* imgWidth + x ];
        x5 = in[y* imgWidth + x + 1];
 
        x6 = in[(y+1)* imgWidth + x - 1];
        x7 = in[(y+1)* imgWidth + x ];
        x8 = in[(y+1)* imgWidth + x + 1];
 
        Gx = x0 + 2*x3 + x6 - x2 - 2 * x5 - x8;
        Gy = x0 + 2 * x1 + x2 - x6 - 2 * x7 - x8;
        out[index] = (sycl::abs(Gx) + sycl::abs(Gy)) / 2;
    }
 
}
 
//CPU soble
void sobel_cpu(Mat srcImg, Mat dstImg, int imgHeight, int imgWidth)
{
    int Gx = 0;
    int Gy = 0;
    for(int i = 1; i < imgHeight-1; i++)
    {
        unsigned char *dataUp = srcImg.ptr<unsigned char>(i-1);
        unsigned char *data = srcImg.ptr<unsigned char>(i);
        unsigned char *dataDown = srcImg.ptr<unsigned char>(i+1);
        unsigned char *out = dstImg.ptr<unsigned char>(i);
        for (int j = 1; j < imgWidth-1; j++)
        {
            Gx = (dataUp[j+1] + 2 * data[j+1] + dataDown[j+1]) - (dataUp[j-1] + 2 * data[j-1] + dataDown[j-1]);
            Gy = (dataUp[j-1] + 2 * dataUp[j] + dataUp[j+1]) - (dataDown[j-1] + 2 * dataDown[j] + dataDown[j+1]);
            out[j] = (abs(Gx) + abs(Gy))/2;
 
        }
    }
 
}
 
int main()
{
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.default_queue();
    //opencv 读图像
 
    Mat grayImg = imread("1.jpg", 0);
 
    int imgWidth = grayImg.cols;
    int imgHeight = grayImg.rows;
 
    // 对gray image 进行去噪
    Mat gaussImg;
    GaussianBlur(grayImg, gaussImg, Size(3,3), 0, 0, BORDER_DEFAULT);
 
    // dst_cpu, dst_gpu
    Mat dst_cpu(imgHeight, imgWidth, CV_8UC1, Scalar(0));
    Mat dst_gpu(imgHeight, imgWidth, CV_8UC1, Scalar(0));
 
 
    //sobel_cpu
    auto start = std::chrono::system_clock::now();
    sobel_cpu(gaussImg, dst_cpu, imgHeight, imgWidth);
    auto end = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    auto dur = double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
    std::cout << "cpu process time: " << 1000*dur << "ms" << std::endl;
	
 
    //申请指针，并将它指向GPU空间
    size_t num = imgHeight * imgWidth * sizeof(unsigned char);
    unsigned char * in_gpu;
 
    unsigned char *out_gpu;

    in_gpu = (unsigned char *)sycl::malloc_device(num, q_ct1);
    out_gpu = (unsigned char *)sycl::malloc_device(num, q_ct1);

    start = std::chrono::system_clock::now();
    //定义grid 和 block的维度
    sycl::range<3> threadsPerBlock(1, 32, 32); // 32x32 = 1024 不能超过1024
    sycl::range<3> blocksPerGrid(
        1, (imgHeight + threadsPerBlock[2] - 1) / threadsPerBlock[1],
        (imgWidth + threadsPerBlock[2] - 1) / threadsPerBlock[2]);

    q_ct1.memcpy(in_gpu, gaussImg.data, num).wait();

    end = std::chrono::system_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    dur = double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
    std::cout << "cpu->gpu copy time: " << 1000*dur << "ms ";
 
 
    start = std::chrono::system_clock::now();

    /*
    DPCT1049:0: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    q_ct1.parallel_for(
        sycl::nd_range<3>(blocksPerGrid * threadsPerBlock, threadsPerBlock),
        [=](sycl::nd_item<3> item_ct1) {
            sobel_gpu(in_gpu, out_gpu, imgHeight, imgWidth, item_ct1);
        });

    end = std::chrono::system_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    dur = double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
    std::cout << " gpu process time: " << 1000*dur << "ms ";
 
    start = std::chrono::system_clock::now();

    q_ct1.memcpy(dst_gpu.data, out_gpu, num).wait();

    end = std::chrono::system_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    dur = double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
    std::cout << " gpu->cpu copy time: " << 1000*dur << "ms" << std::endl;
 
    //显示处理结果
    //imshow("gpu", dst_gpu);
    //imshow("cpu", dst_cpu);
    imwrite("gpu_process.bmp", dst_gpu);
    imwrite("cpu_process.bmp", dst_cpu);

    sycl::free(in_gpu, q_ct1);
    sycl::free(out_gpu, q_ct1);

    return 0;
}
