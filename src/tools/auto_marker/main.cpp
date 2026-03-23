#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include "imageproc/imageproc.hpp"
#include "cuda_utils.hpp"
#include "trt_detector.hpp"

using namespace cv;
using namespace std;
using namespace imgproc;

namespace bfs = boost::filesystem;

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

static inline void check_error(cudaError_t status)
{
    check_error_cuda(status);
}


int main(int argc, char **argv) 
{
    string dir;
    string engine_path = "data/algorithm/best.engine";
    if (argc >= 2)
        dir.assign(argv[1]);
    else
    {
        cout << "usage: auto_marker <images_path> [engine_path]" << endl;
        exit(0);
    }
    if (argc >= 3)
        engine_path.assign(argv[2]);

    string label_path = dir+"_labels";
    bfs::path lpath(label_path);
    if(!bfs::exists(lpath))
    {
        bfs::create_directories(lpath);
    }
    vector<string> picnames = get_filenames(dir);

    // Keep legacy label behavior in this tool: class 0 = ball, class 1 = post.
    TRTDetector detector;
    if (!detector.load(engine_path, 0, 1))
    {
        cerr << "failed to load TensorRT engine: " << engine_path << endl;
        return 1;
    }

    unsigned char *dev_src_;
    unsigned char *dev_sized_;
    float *dev_rgbfp_;
    cudaError_t err;

    const int nw = detector.input_w();
    const int nh = detector.input_h();
    int resize_size = nh*nw*3*sizeof(unsigned char);
    int rgbfp_size = nh*nw*3*sizeof(float);
    err = cudaMalloc((void **)&dev_sized_, resize_size);
    check_error(err);
    err = cudaMalloc((void **)&dev_rgbfp_, rgbfp_size);
    check_error(err);    
    for(int i=0;i<picnames.size();i++)
    {
        cout<<"label: "<<picnames[i]<<endl;
        Mat src = imread(picnames[i]);
        int w = src.size().width;
        int h = src.size().height;
        int src_size = h*w*3*sizeof(unsigned char);
        err = cudaMalloc((void **)&dev_src_, src_size);
        check_error(err);
        err = cudaMemcpy(dev_src_, src.data, src_size, cudaMemcpyHostToDevice);
        check_error(err);
        cudaResizePacked(dev_src_, w, h, dev_sized_, nw, nh);
        cudaBGR2RGBfp(dev_sized_, dev_rgbfp_, nw, nh);

        vector<object_det> ball_dets;
        vector<object_det> post_dets;
        const bool ok = detector.detect(dev_rgbfp_,
                                        w,
                                        h,
                                        ball_dets,
                                        post_dets,
                                        0.5f,
                                        0.4f,
                                        20,
                                        20,
                                        15,
                                        20,
                                        0.3f,
                                        0.45f);

        bfs::path file(picnames[i]);
        string fn = file.filename().string();
        ofstream label(label_path+'/'+fn.substr(0, fn.size()-3)+"txt");

        if (ok)
        {
            for (const auto &det : ball_dets)
            {
                const int x1 = std::max(0, det.x);
                const int y1 = std::max(0, det.y);
                const int x2 = std::min(w, det.x + det.w);
                const int y2 = std::min(h, det.y + det.h);
                const int bw = std::max(0, x2 - x1);
                const int bh = std::max(0, y2 - y1);
                if (bw <= 0 || bh <= 0)
                    continue;

                const float cx = (x1 + bw * 0.5f) / static_cast<float>(w);
                const float cy = (y1 + bh * 0.5f) / static_cast<float>(h);
                const float nw_box = bw / static_cast<float>(w);
                const float nh_box = bh / static_cast<float>(h);
                label << 0 << ' ' << cx << ' ' << cy << ' ' << nw_box << ' ' << nh_box << endl;
            }

            for (const auto &det : post_dets)
            {
                const int x1 = std::max(0, det.x);
                const int y1 = std::max(0, det.y);
                const int x2 = std::min(w, det.x + det.w);
                const int y2 = std::min(h, det.y + det.h);
                const int pw = std::max(0, x2 - x1);
                const int ph = std::max(0, y2 - y1);
                if (pw <= 0 || ph <= 0)
                    continue;

                const float cx = (x1 + pw * 0.5f) / static_cast<float>(w);
                const float cy = (y1 + ph * 0.5f) / static_cast<float>(h);
                const float nw_box = pw / static_cast<float>(w);
                const float nh_box = ph / static_cast<float>(h);
                label << 1 << ' ' << cx << ' ' << cy << ' ' << nw_box << ' ' << nh_box << endl;
            }
        }

        label.close();
        cudaFree(dev_src_);
    }

    detector.release();
    cudaFree(dev_sized_);
    cudaFree(dev_rgbfp_);
    return 0;
}