#ifndef __CAMERA_HPP
#define __CAMERA_HPP

#include <map>
#include <thread>
#include <linux/videodev2.h>
#include <CameraApi.h>
#include "sensor.hpp"
#include "model.hpp"
#include "configuration.hpp"

#ifdef USE_ZED_BACKEND
#include <sl/Camera.hpp>
#endif

class Camera: public Sensor
{
public:
    Camera();
    ~Camera();

    bool start();
    void run();
    void stop();
    bool open();
    void close();

    void set_camera_info(const camera_info &para);
    inline unsigned char *buffer() const
    {
        if (use_mv_ || use_zed_)
        {
            return buffer_;
        }
        else
        {
            return buffers_[num_bufs_].start;
        }
    }

    inline int camera_w() const
    {
        return w_;
    }

    inline int camera_h() const
    {
        return h_;
    }

    inline bool use_mv() const
    {
        return use_mv_;
    }

    inline bool use_zed() const
    {
        return use_zed_;
    }

    bool get_zed_left_camera_param(camera_param &para) const;

    inline int camera_size() const
    {
        if (use_mv_)
        {
            return w_ * h_;
        }
        else if (use_zed_)
        {
            return w_ * h_ * 3;
        }
        else
        {
            return w_ * h_ * 2;
        }
    }

    int timestamp_begin, timestamp_end, time_used;
private:
    struct VideoBuffer
    {
        unsigned char *start;
        size_t offset;
        size_t length;
        size_t bytesused;
        int lagtimems;
    };
    bool use_mv_;
    bool use_zed_;
    std::thread td_;
    VideoBuffer *buffers_;
    v4l2_buffer buf_;
    unsigned int num_bufs_;
    int fd_;
    int w_;
    int h_;
    unsigned char *buffer_;

#ifdef USE_ZED_BACKEND
    sl::Camera zed_;
    sl::Mat zed_image_;
    camera_param zed_left_params_;
    bool zed_calib_valid_;
    float zed_left_hfov_;
    float zed_left_vfov_;
    float zed_left_dfov_;
    float zed_stereo_tx_;
    int zed_calib_w_;
    int zed_calib_h_;
#endif
    tSdkCameraCapbility     tCapability_;
    tSdkFrameHead           sFrameInfo_;
    std::map<std::string, camera_info> camera_infos_;
};

#endif
