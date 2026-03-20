/**
 * @brief 自动标注工具 (auto_marker)
 *
 * 使用 TensorRT YOLOv8 引擎对图片目录进行批量推理，
 * 自动生成 YOLO 格式标注文件 (class cx cy w h，归一化坐标)。
 *
 * 用法: auto_marker <images_dir> [engine_file]
 *   - images_dir:  图片目录路径
 *   - engine_file: TensorRT .engine 文件路径（默认 data/algorithm/best.engine）
 *
 * 类别约定 (与 YOLOv8 训练一致):
 *   0 = post (门柱)
 *   1 = ball (足球)
 */

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include "vision/trt_detector.hpp"

using namespace cv;
using namespace std;

namespace bfs = boost::filesystem;

static void check_error(cudaError_t status)
{
    if (status != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(status));
        exit(1);
    }
}

vector<string> get_filenames(const string& dir)
{
    vector<string> filenames;
    bfs::path path(dir);
    if (!bfs::exists(path))
        return filenames;

    bfs::directory_iterator end_iter;
    for (bfs::directory_iterator iter(path); iter!=end_iter; ++iter)
    {
        if (bfs::is_regular_file(iter->status()))
        {
            string picname=iter->path().string();
            bfs::path file(picname);

            if(file.extension().string()==".png"||file.extension().string()==".jpg")
                filenames.push_back(picname);
        }

        if (bfs::is_directory(iter->status()))
        {
            vector<string> temp = get_filenames(iter->path().string());
            filenames.insert(filenames.end(), temp.begin(), temp.end());
        }
    }
    return filenames;
}


int main(int argc, char **argv)
{
    string dir;
    string engine_file = "data/algorithm/best.engine";

    if (argc >= 2)
        dir.assign(argv[1]);
    else
    {
        cout << "Usage: auto_marker <images_dir> [engine_file]" << endl;
        exit(0);
    }
    if (argc >= 3)
        engine_file.assign(argv[2]);

    string label_path = dir + "_labels";
    bfs::path lpath(label_path);
    if (!bfs::exists(lpath))
    {
        bfs::create_directories(lpath);
    }

    vector<string> picnames = get_filenames(dir);
    cout << "Found " << picnames.size() << " images in " << dir << endl;

    // ---- 加载 TensorRT 引擎 ----
    TRTDetector detector;
    // 类别: 0=post, 1=ball（与 YOLOv8 训练 yaml 一致）
    int ball_id = 1, post_id = 0;
    if (!detector.load(engine_file, 2))
    {
        cerr << "Failed to load TensorRT engine: " << engine_file << endl;
        return 1;
    }
    cout << "Engine loaded: " << engine_file
         << " (input " << detector.net_w() << "x" << detector.net_h() << ")" << endl;

    // ---- 检测阈值与过滤参数（与原 auto_marker 一致）----
    float ball_thresh = 0.5f;
    float post_thresh = 0.4f;
    int min_ball_w = 20, min_ball_h = 20;
    int min_post_w = 15, min_post_h = 20;
    float d_w_h = 0.3f;  // 球宽高比容差

    unsigned char *dev_src_ = nullptr;
    cudaError_t err;

    for (size_t i = 0; i < picnames.size(); i++)
    {
        cout << "label: " << picnames[i] << endl;
        Mat src = imread(picnames[i]);
        if (src.empty())
        {
            cerr << "  Skip (cannot read): " << picnames[i] << endl;
            continue;
        }
        int w = src.cols;
        int h = src.rows;
        int src_size = h * w * 3 * sizeof(unsigned char);

        err = cudaMalloc((void **)&dev_src_, src_size);
        check_error(err);
        err = cudaMemcpy(dev_src_, src.data, src_size, cudaMemcpyHostToDevice);
        check_error(err);

        // ---- TensorRT 检测 ----
        vector<object_det> ball_dets, post_dets;
        detector.detect(dev_src_, w, h,
                        ball_dets, post_dets,
                        ball_id, post_id,
                        ball_thresh, post_thresh,
                        min_ball_w, min_ball_h,
                        min_post_w, min_post_h,
                        d_w_h);

        // ---- 写入 YOLO 格式标注文件 ----
        bfs::path file(picnames[i]);
        string fn = file.filename().string();
        ofstream label(label_path + '/' + fn.substr(0, fn.size() - 3) + "txt");

        // 球: class_id=1, 输出归一化 cx cy w h
        for (auto &det : ball_dets)
        {
            float norm_cx = (det.x + det.w / 2.0f) / w;
            float norm_cy = (det.y + det.h / 2.0f) / h;
            float norm_w  = det.w / (float)w;
            float norm_h  = det.h / (float)h;
            label << ball_id << ' ' << norm_cx << ' ' << norm_cy
                  << ' ' << norm_w << ' ' << norm_h << endl;
        }

        // 门柱: class_id=0
        for (auto &det : post_dets)
        {
            float norm_cx = (det.x + det.w / 2.0f) / w;
            float norm_cy = (det.y + det.h / 2.0f) / h;
            float norm_w  = det.w / (float)w;
            float norm_h  = det.h / (float)h;
            label << post_id << ' ' << norm_cx << ' ' << norm_cy
                  << ' ' << norm_w << ' ' << norm_h << endl;
        }

        label.close();
        cudaFree(dev_src_);
        dev_src_ = nullptr;
    }

    detector.release();
    cout << "Done. Labels saved to: " << label_path << endl;
    return 0;
}