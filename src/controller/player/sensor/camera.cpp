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
#include "logger.hpp"
#include "core/clock.hpp"

#ifdef USE_ZED_BACKEND
#include <opencv2/opencv.hpp>
#endif

using namespace std;

Camera::Camera(): Sensor("camera")
{
    // Default states; will be overwritten in open()
    use_mv_ = false;
    use_zed_ = false;
    buffers_ = nullptr;
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
    if (!use_mv_)
    {
        return;
    }

    for (auto &item : camera_infos_)
    {
        if (item.second.id == para.id)
        {
            item.second.value = para.value;

            switch (para.id)
            {
                case 1:
                    CameraSetAnalogGain(fd_, para.value);
                    break;

                case 2:
                    CameraSetExposureTime(fd_, para.value * 1000);
                    break;

                default:
                    break;
            }

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

                    cv::Mat resized_bgra;
                    cv::resize(src_bgra, resized_bgra, cv::Size(w_, h_), 0, 0, cv::INTER_LINEAR);
                    cv::cvtColor(resized_bgra, dst_bgr, cv::COLOR_BGRA2BGR);
                }
                else if (channels == 3)
                {
                    cv::Mat src_c3(
                        src_h, src_w, CV_8UC3,
                        zed_image_.getPtr<sl::uchar3>(sl::MEM::CPU),
                        static_cast<size_t>(zed_image_.getStepBytes(sl::MEM::CPU)));
                    cv::resize(src_c3, dst_bgr, cv::Size(w_, h_), 0, 0, cv::INTER_LINEAR);
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

    if (use_mv_)
    {
        while (is_alive_)
        {
            timestamp_begin = CLOCK->get_timestamp();
            int t1 = sFrameInfo_.uiTimeStamp;
            CameraSdkStatus status = CameraGetImageBuffer(fd_, &sFrameInfo_, &buffer_, 1000);
            timestamp_end = CLOCK->get_timestamp();
            time_used = (sFrameInfo_.uiTimeStamp-t1)*0.1;
            if (status == CAMERA_STATUS_SUCCESS)
            {
                notify(SENSOR_CAMERA);
                CameraReleaseImageBuffer(fd_, buffer_);
            }
            usleep(1000);
        }
    }
    else
    {
        buf_.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf_.memory = V4L2_MEMORY_MMAP;

        while (is_alive_)
        {

            if (ioctl(fd_, VIDIOC_DQBUF, &buf_) == -1)
            {
                LOG(LOG_ERROR) << "VIDIOC_DQBUF failed."<<endll;
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
        return;
    }

    if (use_mv_)
    {
        if (is_open_)
        {
            CameraUnInit(fd_);
        }
    }
    else
    {
        if (is_open_)
        {
            enum v4l2_buf_type type;
            type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

            if (ioctl(fd_, VIDIOC_STREAMOFF, &type))
            {
                LOG(LOG_ERROR) << "VIDIOC_STREAMOFF error"<<endll;
                return;
            }

            for (num_bufs_ = 0; num_bufs_ < 4; num_bufs_++)
            {
                munmap((void *)(buffers_[num_bufs_].start), buffers_[num_bufs_].length);
                buffers_[num_bufs_].start = nullptr;
            }

            free(buffers_);
            buffers_ = nullptr;
            ::close(fd_);
        }
    }
}

bool Camera::open()
{
    int                     iCameraCounts = 1;
    int                     iStatus = -1;
    tSdkCameraDevInfo       tCameraEnumList;
    use_mv_ = true;
    use_zed_ = false;
    buffer_ = nullptr;
    CameraSdkInit(1);
    iStatus = CameraEnumerateDevice(&tCameraEnumList, &iCameraCounts);

    if (iCameraCounts == 0)
    {
        use_mv_ = false;
        LOG(LOG_WARN) << "open MV camera failed, try ZED-mini backend..." << endll;

#ifdef USE_ZED_BACKEND
        // Keep Vision expected resolution (image.width/height) and resize ZED frames in run().
        const int target_w = CONF->get_config_value<int>("image.width");
        const int target_h = CONF->get_config_value<int>("image.height");
        w_ = target_w;
        h_ = target_h;
    zed_calib_valid_ = false;

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
            init_parameters.camera_resolution = sl::RESOLUTION::VGA;
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

                const float sx = (zed_calib_w_ > 0) ? (static_cast<float>(w_) / static_cast<float>(zed_calib_w_)) : 1.0f;
                const float sy = (zed_calib_h_ > 0) ? (static_cast<float>(h_) / static_cast<float>(zed_calib_h_)) : 1.0f;

                zed_left_params_.fx = left_cam.fx * sx;
                zed_left_params_.fy = left_cam.fy * sy;
                zed_left_params_.cx = left_cam.cx * sx;
                zed_left_params_.cy = left_cam.cy * sy;
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
                              << " target=" << w_ << "x" << h_
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
            LOG(LOG_ERROR) << "ZED backend open failed: " << static_cast<int>(returned_state) << endll;
        }
#endif

        LOG(LOG_ERROR) << "MV camera failed and ZED backend unavailable, fallback to V4L2..." << endll;
    }

    if (use_mv_)
    {
        iStatus = CameraInit(&tCameraEnumList, -1, PARAMETER_TEAM_DEFAULT, &fd_);

        if (iStatus != CAMERA_STATUS_SUCCESS)
        {
            return false;
        }

        CameraGetCapability(fd_, &tCapability_);
        CameraSetAeState(fd_, false);
        CameraSetAnalogGain(fd_, camera_infos_["exposure_gain"].value);
        CameraSetExposureTime(fd_, camera_infos_["exposure_time"].value * 1000);
        CameraSetImageResolution(fd_, &(tCapability_.pImageSizeDesc[0]));
        w_ = tCapability_.pImageSizeDesc[0].iWidth;
        h_ = tCapability_.pImageSizeDesc[0].iHeight;
        CameraPlay(fd_);
    }
    else
    {
        fd_ = ::open(CONF->get_config_value<string>("image.dev_name").c_str(), O_RDWR, 0);

        if (fd_ < 0)
        {
            LOG(LOG_ERROR) << "open camera: " + CONF->get_config_value<string>("image.dev_name") + " failed"<<endll;
            return false;
        }

        w_ = 640;
        h_ = 480;
        v4l2_format fmt;
        fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
        fmt.fmt.pix.width = w_;
        fmt.fmt.pix.height = h_;
        fmt.fmt.pix.field = V4L2_FIELD_INTERLACED;

        if (ioctl(fd_, VIDIOC_S_FMT, &fmt) < 0)
        {
            LOG(LOG_ERROR) << "set format failed"<<endll;
            return false;
        }

        if (ioctl(fd_, VIDIOC_G_FMT, &fmt) < 0)
        {
            LOG(LOG_ERROR) << "get format failed"<<endll;
            return false;
        }

        v4l2_requestbuffers req;
        req.count = 4;
        req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        req.memory = V4L2_MEMORY_MMAP;

        if (ioctl(fd_, VIDIOC_REQBUFS, &req) == -1)
        {
            LOG(LOG_ERROR) << "request buffer error"<<endll;
            return false;
        }

        buffers_ = (VideoBuffer *)calloc(req.count, sizeof(VideoBuffer));

        for (num_bufs_ = 0; num_bufs_ < req.count; num_bufs_++)
        {
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
