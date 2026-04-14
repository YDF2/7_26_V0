#include <sys/mman.h>
#include <errno.h>
#include <functional>
#include "camera.hpp"
#include "parser/parser.hpp"
#include "configuration.hpp"
#include "class_exception.hpp"
#include <sstream>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <cstring>
#include "logger.hpp"
#include "core/clock.hpp"

#ifdef USE_ZED_BACKEND
#include <opencv2/opencv.hpp>
#endif

using namespace std;

Camera::Camera(): Sensor("camera")
{
    // Default states; will be overwritten in open()
    use_zed_ = false;
    buffers_ = nullptr;
    buffer_count_ = 0;
    num_bufs_ = 0;
    fd_ = -1;
    w_ = 0;
    h_ = 0;
    buffer_ = nullptr;

#ifdef USE_ZED_BACKEND
    zed_calib_valid_ = false;
    zed_left_hfov_ = 0.0f;
    zed_left_vfov_ = 0.0f;
    zed_left_dfov_ = 0.0f;
    zed_stereo_tx_ = 0.0f;
    zed_calib_w_ = 0;
    zed_calib_h_ = 0;
#endif

    parser::parse(CONF->get_config_value<string>(CONF->player() + ".camera_info_file"), camera_infos_);
}

bool Camera::get_zed_left_camera_param(camera_param &para) const
{
#ifdef USE_ZED_BACKEND
    if (!zed_calib_valid_)
    {
        return false;
    }
    para = zed_left_params_;
    return true;
#else
    (void)para;
    return false;
#endif
}

bool Camera::start()
{
    if (!this->open())
    {
        return false;
    }

    td_ = std::move(thread(bind(&Camera::run, this)));

    return true;
}

void Camera::set_camera_info(const camera_info &para)
{
    // Keep config cache update for debug tools; no runtime camera control without MV.
    for (auto &item : camera_infos_)
    {
        if (item.second.id == para.id)
        {
            item.second.value = para.value;
            break;
        }
    }
}

void Camera::run()
{
    is_alive_ = true;

#ifdef USE_ZED_BACKEND
    if (use_zed_)
    {
        if (buffer_ == nullptr || w_ <= 0 || h_ <= 0)
        {
            LOG(LOG_ERROR) << "ZED backend not initialized correctly (buffer/w/h)" << endll;
            return;
        }

        cv::Mat dst_bgr(h_, w_, CV_8UC3, buffer_);
        while (is_alive_)
        {
            timestamp_begin = CLOCK->get_timestamp();

            sl::ERROR_CODE returned_state = zed_.grab();
            if (returned_state <= sl::ERROR_CODE::SUCCESS)
            {
                zed_.retrieveImage(zed_image_, sl::VIEW::LEFT, sl::MEM::CPU);

                const int src_w = static_cast<int>(zed_image_.getWidth());
                const int src_h = static_cast<int>(zed_image_.getHeight());
                const int channels = static_cast<int>(zed_image_.getChannels());

                if (channels == 4)
                {
                    cv::Mat src_bgra(
                        src_h, src_w, CV_8UC4,
                        zed_image_.getPtr<sl::uchar4>(sl::MEM::CPU),
                        static_cast<size_t>(zed_image_.getStepBytes(sl::MEM::CPU)));
                    if (src_w == w_ && src_h == h_)
                    {
                        cv::cvtColor(src_bgra, dst_bgr, cv::COLOR_BGRA2BGR);
                    }
                    else
                    {
                        cv::Mat resized_bgra;
                        cv::resize(src_bgra, resized_bgra, cv::Size(w_, h_), 0, 0, cv::INTER_LINEAR);
                        cv::cvtColor(resized_bgra, dst_bgr, cv::COLOR_BGRA2BGR);
                    }
                }
                else if (channels == 3)
                {
                    cv::Mat src_c3(
                        src_h, src_w, CV_8UC3,
                        zed_image_.getPtr<sl::uchar3>(sl::MEM::CPU),
                        static_cast<size_t>(zed_image_.getStepBytes(sl::MEM::CPU)));
                    if (src_w == w_ && src_h == h_)
                    {
                        src_c3.copyTo(dst_bgr);
                    }
                    else
                    {
                        cv::resize(src_c3, dst_bgr, cv::Size(w_, h_), 0, 0, cv::INTER_LINEAR);
                    }
                }
                else
                {
                    // Unexpected channel count, skip notify.
                    continue;
                }

                timestamp_end = CLOCK->get_timestamp();
                time_used = static_cast<int>(timestamp_end - timestamp_begin);
                notify(SENSOR_CAMERA);
            }
            else
            {
                usleep(1000);
            }
        }
        return;
    }
#endif

    buf_.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf_.memory = V4L2_MEMORY_MMAP;

    while (is_alive_)
    {
        if (ioctl(fd_, VIDIOC_DQBUF, &buf_) == -1)
        {
            LOG(LOG_ERROR) << "VIDIOC_DQBUF failed."<<endll;
            break;
        }

        if (buf_.index >= buffer_count_)
        {
            LOG(LOG_ERROR) << "V4L2 buffer index out of range: " << buf_.index
                           << " >= " << buffer_count_ << endll;
            break;
        }

        num_bufs_ = buf_.index;
        buffers_[num_bufs_].bytesused = buf_.bytesused;
        buffers_[num_bufs_].length = buf_.length;
        buffers_[num_bufs_].offset = buf_.m.offset;
        notify(SENSOR_CAMERA);

        if (ioctl(fd_, VIDIOC_QBUF, &buf_) == -1)
        {
            LOG(LOG_ERROR) << "VIDIOC_QBUF error"<<endll;
            break;
        }

        num_bufs_ = buf_.index;

        usleep(10000);
    }
}

void Camera::stop()
{
    is_alive_ = false;
    sleep(1);
    this->close();
    is_open_ = false;
}

void Camera::close()
{
    if (use_zed_)
    {
#ifdef USE_ZED_BACKEND
        if (is_open_)
        {
            zed_.close();
            if (buffer_)
            {
                free(buffer_);
                buffer_ = nullptr;
            }
        }
#endif
        use_zed_ = false;
        return;
    }

    if (is_open_)
    {
        enum v4l2_buf_type type;
        type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

        if (fd_ >= 0 && ioctl(fd_, VIDIOC_STREAMOFF, &type))
        {
            LOG(LOG_ERROR) << "VIDIOC_STREAMOFF error"<<endll;
        }

        for (unsigned int i = 0; i < buffer_count_; ++i)
        {
            if (buffers_ && buffers_[i].start && buffers_[i].start != MAP_FAILED)
            {
                munmap((void *)(buffers_[i].start), buffers_[i].length);
                buffers_[i].start = nullptr;
            }
        }

        free(buffers_);
        buffers_ = nullptr;
        buffer_count_ = 0;
        num_bufs_ = 0;

        if (fd_ >= 0)
        {
            ::close(fd_);
            fd_ = -1;
        }
    }
}

bool Camera::open()
{
    use_zed_ = false;
    buffers_ = nullptr;
    buffer_count_ = 0;
    num_bufs_ = 0;
    fd_ = -1;
    buffer_ = nullptr;
    const int target_w = CONF->get_config_value<int>("image.width");
    const int target_h = CONF->get_config_value<int>("image.height");
    w_ = target_w;
    h_ = target_h;

#ifdef USE_ZED_BACKEND
    LOG(LOG_INFO) << "Camera: try ZED-mini backend first." << endll;
    zed_calib_valid_ = false;

    // Capture in HD720 and keep native frame in camera buffer.
    const int zed_capture_w = 1280;
    const int zed_capture_h = 720;
    w_ = zed_capture_w;
    h_ = zed_capture_h;

    buffer_ = (unsigned char *)malloc(static_cast<size_t>(w_) * static_cast<size_t>(h_) * 3);
    if (!buffer_)
    {
        LOG(LOG_ERROR) << "ZED backend: malloc failed" << endll;
    }
    else
    {
        sl::InitParameters init_parameters;
        init_parameters.depth_mode = sl::DEPTH_MODE::NONE;
        init_parameters.coordinate_units = sl::UNIT::METER;

        init_parameters.camera_resolution = sl::RESOLUTION::HD720;
        init_parameters.camera_fps = 60;

        auto returned_state = zed_.open(init_parameters);
        if (returned_state == sl::ERROR_CODE::SUCCESS)
        {
            auto camera_info = zed_.getCameraInformation();
            auto calib_params = camera_info.camera_configuration.calibration_parameters;
            auto left_cam = calib_params.left_cam;

            zed_calib_w_ = static_cast<int>(left_cam.image_size.width);
            zed_calib_h_ = static_cast<int>(left_cam.image_size.height);
            if (zed_calib_w_ <= 0 || zed_calib_h_ <= 0)
            {
                zed_calib_w_ = static_cast<int>(camera_info.camera_configuration.resolution.width);
                zed_calib_h_ = static_cast<int>(camera_info.camera_configuration.resolution.height);
            }

            const float src_w = static_cast<float>(zed_calib_w_);
            const float src_h = static_cast<float>(zed_calib_h_);
            const float target_aspect = static_cast<float>(target_w) / static_cast<float>(target_h);

            float crop_w = src_w;
            float crop_h = src_h;
            if (src_w / src_h > target_aspect)
            {
                crop_w = src_h * target_aspect;
            }
            else
            {
                crop_h = src_w / target_aspect;
            }

            const float crop_x = 0.5f * (src_w - crop_w);
            const float crop_y = 0.5f * (src_h - crop_h);
            const float sx = static_cast<float>(target_w) / crop_w;
            const float sy = static_cast<float>(target_h) / crop_h;

            zed_left_params_.fx = left_cam.fx * sx;
            zed_left_params_.fy = left_cam.fy * sy;
            zed_left_params_.cx = (left_cam.cx - crop_x) * sx;
            zed_left_params_.cy = (left_cam.cy - crop_y) * sy;
            zed_left_params_.k1 = left_cam.disto[0];
            zed_left_params_.k2 = left_cam.disto[1];
            zed_left_params_.p1 = left_cam.disto[2];
            zed_left_params_.p2 = left_cam.disto[3];

            zed_left_hfov_ = left_cam.h_fov;
            zed_left_vfov_ = left_cam.v_fov;
            zed_left_dfov_ = left_cam.d_fov;
            auto stereo_t = calib_params.stereo_transform.getTranslation();
            zed_stereo_tx_ = stereo_t.x;
            zed_calib_valid_ = true;

            LOG(LOG_INFO) << "ZED calibration(left): src=" << zed_calib_w_ << "x" << zed_calib_h_
                          << " capture=" << w_ << "x" << h_
                          << " target=" << target_w << "x" << target_h
                          << " crop=" << static_cast<int>(crop_w) << "x" << static_cast<int>(crop_h)
                          << " crop_xy=" << static_cast<int>(crop_x) << "," << static_cast<int>(crop_y)
                          << " fx=" << zed_left_params_.fx
                          << " fy=" << zed_left_params_.fy
                          << " cx=" << zed_left_params_.cx
                          << " cy=" << zed_left_params_.cy
                          << " k1=" << zed_left_params_.k1
                          << " k2=" << zed_left_params_.k2
                          << " p1=" << zed_left_params_.p1
                          << " p2=" << zed_left_params_.p2
                          << " h_fov=" << zed_left_hfov_
                          << " v_fov=" << zed_left_vfov_
                          << " d_fov=" << zed_left_dfov_
                          << " tx=" << zed_stereo_tx_ << endll;

            use_zed_ = true;
            is_open_ = true;
            return true;
        }

        free(buffer_);
        buffer_ = nullptr;
        LOG(LOG_WARN) << "ZED backend open failed: " << static_cast<int>(returned_state)
                      << ", fallback to V4L2." << endll;
    }
#else
    LOG(LOG_WARN) << "ZED backend is not compiled, fallback to V4L2." << endll;
#endif

    // Restore configured target size for V4L2 path.
    w_ = target_w;
    h_ = target_h;

    std::string dev_name = CONF->get_config_value<string>("image.dev_name");
    LOG(LOG_INFO) << "Camera: open V4L2 device " << dev_name << endll;
    fd_ = ::open(dev_name.c_str(), O_RDWR, 0);

    if (fd_ < 0)
    {
        LOG(LOG_ERROR) << "open camera: " + dev_name + " failed"<<endll;
        return false;
    }

    v4l2_format fmt;
    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
    fmt.fmt.pix.width = w_;
    fmt.fmt.pix.height = h_;
    fmt.fmt.pix.field = V4L2_FIELD_INTERLACED;

    if (ioctl(fd_, VIDIOC_S_FMT, &fmt) < 0)
    {
        LOG(LOG_ERROR) << "set format failed"<<endll;
        ::close(fd_);
        fd_ = -1;
        return false;
    }

    if (ioctl(fd_, VIDIOC_G_FMT, &fmt) < 0)
    {
        LOG(LOG_ERROR) << "get format failed"<<endll;
        ::close(fd_);
        fd_ = -1;
        return false;
    }

    w_ = static_cast<int>(fmt.fmt.pix.width);
    h_ = static_cast<int>(fmt.fmt.pix.height);

    v4l2_requestbuffers req;
    memset(&req, 0, sizeof(req));
    req.count = 4;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;

    if (ioctl(fd_, VIDIOC_REQBUFS, &req) == -1)
    {
        LOG(LOG_ERROR) << "request buffer error"<<endll;
        ::close(fd_);
        fd_ = -1;
        return false;
    }

    if (req.count == 0)
    {
        LOG(LOG_ERROR) << "request buffer returned 0 buffers" << endll;
        ::close(fd_);
        fd_ = -1;
        return false;
    }

    buffer_count_ = req.count;
    buffers_ = (VideoBuffer *)calloc(buffer_count_, sizeof(VideoBuffer));
    if (!buffers_)
    {
        LOG(LOG_ERROR) << "calloc V4L2 buffers failed" << endll;
        ::close(fd_);
        fd_ = -1;
        buffer_count_ = 0;
        return false;
    }

    for (num_bufs_ = 0; num_bufs_ < buffer_count_; num_bufs_++)
    {
        memset(&buf_, 0, sizeof(buf_));
        buf_.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf_.memory = V4L2_MEMORY_MMAP;
        buf_.index = num_bufs_;

        if (ioctl(fd_, VIDIOC_QUERYBUF, &buf_) == -1)
        {
            LOG(LOG_ERROR) << "query buffer error"<<endll;
            return false;
        }

        buffers_[num_bufs_].length = buf_.length;
        buffers_[num_bufs_].offset = (size_t) buf_.m.offset;
        buffers_[num_bufs_].start = (unsigned char *)mmap(NULL, buf_.length, PROT_READ | PROT_WRITE,
                                    MAP_SHARED, fd_, buf_.m.offset);

        if (buffers_[num_bufs_].start == MAP_FAILED)
        {
            int err = errno;
            LOG(LOG_ERROR) << "buffer map error: " << err << endll;
            return false;
        }

        if (ioctl(fd_, VIDIOC_QBUF, &buf_) == -1)
        {
            LOG(LOG_ERROR) << "VIDIOC_QBUF error"<<endll;
            return false;
        }
    }

    enum v4l2_buf_type type;
    type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

    if (ioctl(fd_, VIDIOC_STREAMON, &type) == -1)
    {
        LOG(LOG_ERROR) << "VIDIOC_STREAMON error"<<endll;
        return false;
    }

    is_open_ = true;
    return true;
}


Camera::~Camera()
{
    if (td_.joinable())
    {
        td_.join();
    }
}
