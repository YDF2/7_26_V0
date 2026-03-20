#ifndef __IMAGE_PROC_HPP
#define __IMAGE_PROC_HPP

#include <memory>
#include <string>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "model.hpp"

namespace imgproc
{
    void cudaYUYV2YUV(unsigned char *in, unsigned char *out, int w, int h);
    void cudaYUYV2BGR(unsigned char *in, unsigned char *out, int w, int h);
    void cudaBGR2RGBfp(unsigned char *bgr, float *rgbfp, int w, int h);
    void cudaBGR2YUV422(unsigned char *bgr, unsigned char *yuv422, int w, int h);
    void cudaBayer2BGR(unsigned char *bayer, unsigned char *bgr, int w, int h, float sat, float rgain, float ggain, float bgain);
    void cudaBGR2HSV(unsigned char *bgr, unsigned char *hsv, int w, int h);

    void cudaResizePacked(float *in, int iw, int ih, float *sized, int ow, int oh);
    void cudaResizePacked(unsigned char *in, int iw, int ih, unsigned char *sized, int ow, int oh);
    void cudaUndistored(unsigned char *in, unsigned char *out, float *pCamK, float *pDistort, float *pInvNewCamK, float* pMapx, float* pMapy, int w, int h, int c);
};
#endif
