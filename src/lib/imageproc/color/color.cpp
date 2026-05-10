#include "color.hpp"
#include <fstream>
#include <cuda_runtime.h>
#include <cstdio>
#include <cassert>

// check_error 本地定义（替代 darknet/cuda.h 中的版本）
static void check_error(cudaError_t status)
{
    if (status != cudaSuccess)
    {
        const char *s = cudaGetErrorString(status);
        fprintf(stderr, "CUDA Error: %s\n", s);
        assert(0);
    }
}

using namespace std;
using namespace cv;

const uint8_t BIT[8] =
{
    0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80
};

ColorTable::ColorTable(bool debug): debug_(debug)
{
    cube_size = 256;
    table_size = cube_size*cube_size*cube_size*sizeof(uint8_t);
    data = new uint8_t[table_size];
    cudaError err = cudaMalloc((void**)&dev_data, table_size);
    check_error(err);
    if(debug)
        hist_data = new uint8_t[table_size*8];
    Clear();
}

ColorTable::~ColorTable()
{
    delete []data;
    cudaFree(dev_data);
    if(debug_)
        delete []hist_data;
}

void ColorTable::LoadFromFile(string filename)
{
    fstream CTfile;
    int i, j, k;
    CTfile.open(filename.c_str(), ios::in | ios::binary);
    CTfile.read((char*)data, table_size);
    CTfile.close();
    cudaError err = cudaMemcpy(dev_data, data, table_size, cudaMemcpyHostToDevice);
}

void ColorTable::Clear()
{
    memset(data, 0, table_size);
    if(debug_)
        memset(hist_data, 0, table_size*8);
    cudaError err = cudaMemcpy(dev_data, data, table_size, cudaMemcpyHostToDevice);
    check_error(err);
}

void ColorTable::Update(Mat frame, int x, int y, int width, int height, int color_index)
{
    int i, j;
    for (i = y ; i <= y + height ; i++)
    {
        for (j = x; j <= x + width ; j++)
        {
            Vec3b s = frame.at<Vec3b>(i, j);
            int Y=s[0], U=s[1], V=s[2];
            uint8_t &hd = *(hist_data+Y*256*256*8+U*256*8+V*8+color_index);
            if (hd < 250) hd++;
        }
    }
}

void ColorTable::GenerateColorTable()
{
    int i, j, k, l;
    int index_most, index_sec;
    int num_most, num_sec;
    float w;
    int temp;

    for (i = 0; i < 256; i++)
    {
        for (j = 0; j < 256; j++)
        {
            for (k = 0; k < 256; k++)
            {
                temp = 0;
                for (l = 0; l < 8; l++)
                    temp += *(hist_data+i*256*256*8+j*256*8+k*8+l);
                if (temp == 0)
                    continue;
                index_most = 0;
                num_most = *(hist_data+i*256*256*8+j*256*8+k*8+0);
                for (l = 1; l < 8; l++)
                {
                    if (*(hist_data+i*256*256*8+j*256*8+k*8+l) > num_most)
                    {
                        index_most = l;
                        num_most = *(hist_data+i*256*256*8+j*256*8+k*8+l);
                    }
                }
                for (l = 0; l < 8; l++)
                {
                    if (l != index_most)
                    {
                        index_sec = l;
                        num_sec = *(hist_data+i*256*256*8+j*256*8+k*8+l);
                        break;
                    }
                }
                for (l = 0; l < 8; l++)
                {
                    if (l == index_most)
                        continue;
                    if (*(hist_data+i*256*256*8+j*256*8+k*8+l) > num_sec)
                    {
                        index_sec = l;
                        num_sec = *(hist_data+i*256*256*8+j*256*8+k*8+l);
                    }
                }
                w = num_sec * 1.0 / (num_most + num_sec);

                if (w > 0.2)
                    *(data+i*256*256+j*256+k) = BIT[index_most] + BIT[index_sec];
                else
                    *(data+i*256*256+j*256+k) = BIT[index_most];
            }
        }
    }
    cudaError err = cudaMemcpy(dev_data, data, table_size, cudaMemcpyHostToDevice);
    check_error(err);
}

void ColorTable::ExportColorTable(string filename)
{
    GenerateColorTable();
    fstream CTfile;
    int i, j, k;
    CTfile.open(filename.c_str(), ios::out | ios::binary);
    CTfile.write((const char*)data, table_size);
    CTfile.close();
}

Color::Color()
{
    ct_ = make_shared<ColorTable>();
    classifiers_[COLOR_GREEN] = ColorTableClassifier(ct_, 0x01);
    classifiers_[COLOR_WHITE] = ColorTableClassifier(ct_, 0x02);
}

void Color::UpdateColor(int x, int y, int width, int height, Color::color_type color)
{
    if (width <= 5 && height <= 5) //clear
    {
        std::cout << "Start Clear ..." << std::endl;
        ct_->Clear();
        ct_->GenerateColorTable();
        //generator.GenerateAndExport(CONF.cfgs().getConfigValue<string>(CONF.player() + ".colortable_file"));
        //std::cout << "Clear Colortab :" << CONF.cfgs().getConfigValue<string>(CONF.player() + ".colortable_file") << std::endl;
    }
    else
    {
        std::cout << "Start Update ..." << std::endl;

        //ct_->Update(hsv_, x, y, width, height, classifiers_[color].index_);
        ct_->GenerateColorTable();
        //std::cout << (unsigned int)_classifiers[color]._ctbit << _classifiers[color]._index << std::endl;
        //generator.GenerateAndExport(CONF.cfgs().getConfigValue<string>(CONF.player() + ".colortable_file"));
        //_ct.Load(generator.CT_result);
        //std::cout << "Update Colortab :" << CONF.cfgs().getConfigValue<string>(CONF.player() + ".colortable_file") << std::endl;
        //export_to_file(CONF.cfgs().getConfigValue<string>(CONF.player() + ".colorprovider_file"));
    }
}
