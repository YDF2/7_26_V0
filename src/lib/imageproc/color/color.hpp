#ifndef __COLOR_HPP
#define __COLOR_HPP

#include <vector>
#include <cstdint>
#include <string>
#include <memory>
#include <algorithm>
#include <opencv2/opencv.hpp>

class ColorTable
{
public:
    uint8_t *data;
    uint8_t *dev_data;
    uint8_t *hist_data;

public:
    ColorTable(bool debug=false);
    ~ColorTable();
    void Clear();

    void Update(cv::Mat frame, int x, int y, int width, int height, int color_index);
    void LoadFromFile(std::string filename);
    void GenerateColorTable();
    void ExportColorTable(std::string filename);
    
private:
    int cube_size;
    unsigned int table_size;
    bool debug_;
};

struct ColorTableClassifier
{
    std::shared_ptr<ColorTable> ct_;
    uint8_t ctbit_;
    int index_;

    ColorTableClassifier() {}
    ColorTableClassifier(std::shared_ptr<ColorTable> ct, uint8_t bit)
        : ct_(ct), ctbit_(bit)
    {
        auto b = ctbit_;
        index_ = -1;

        while (b != 0)
        {
            b = b >> 1;
            ++ index_;
        }
    }
    ColorTableClassifier &operator=(const ColorTableClassifier &cc)
    {
        ct_ = cc.ct_;
        ctbit_ = cc.ctbit_;
        index_ = cc.index_;
    }

    bool check(const cv::Vec3b &s)
    {
        return (*((ct_->data)+s[0]*256*256+s[1]*256+s[2]) & ctbit_);
    }
};

class Color
{
public:
    enum color_type
    {
        COLOR_UNKNOW,
        COLOR_GREEN,
        COLOR_WHITE
    };

public:
    Color();
    void UpdateColor(int x, int y, int width, int height, color_type color);
    /*
    bool CheckColor(int x, int y, color_type color)
    {
        if(color == COLOR_WHITE)
        {
            return (!classifiers_[color].check(hsv_.at<cv::Vec3b>(y, x)));
        }
        else
            return classifiers_[color].check(hsv_.at<cv::Vec3b>(y, x));
    }
    */
    std::shared_ptr<ColorTable> ct_;

private:
    std::map<color_type, ColorTableClassifier> classifiers_;
};

#endif
