#include <list>
#include "player.hpp"
#include "task/look_task.hpp"
#include "task/walk_task.hpp"
#include "tcp.hpp"
#include "server/server.hpp"
#include "engine/action/action_engine.hpp"
#include "engine/walk/walk_engine.hpp"
#include "engine/scan/scan_engine.hpp"

using namespace robot;
using namespace std;
using namespace motion;

void Player::play_with_remote()
{
    remote_data rdata = SERVER->rmt_data();
    if (rdata.type == WALK_DATA)
    {
        float x, y, d;
        bool e;
        memcpy(&x, rdata.data.c_str(), float_size);
        memcpy(&y, rdata.data.c_str() + float_size, float_size);
        memcpy(&d, rdata.data.c_str() + 2 * float_size, float_size);
        memcpy(&e, rdata.data.c_str() + 3 * float_size, bool_size);
        WE->set_params(x, y, d, e);
    }
    else if (rdata.type == ACT_DATA)
    {
        int blksz = int_size + 5 * sizeof(robot_pose);
        int pos_count = rdata.size / blksz;

        int act_t;
        robot_pose temp_pose;
        map<robot_motion, robot_pose> posesinfo;
        vector< map<robot_motion, robot_pose> > poses;
        vector<int> pos_times;

        for (int i = 0; i < pos_count; i++)
        {
            memcpy(&act_t, rdata.data.c_str() + i * blksz, int_size);
            posesinfo.clear();
            int j = 0;

            for (auto nmm : name_motion_map)
            {
                if (nmm.second != MOTION_HEAD && nmm.second != MOTION_NONE)
                {
                    memcpy(&temp_pose, rdata.data.c_str() + i * blksz + int_size + j * sizeof(robot_pose), sizeof(robot_pose));
                    posesinfo[nmm.second] = temp_pose;
                    j++;
                }
            }

            pos_times.push_back(act_t);
            poses.push_back(posesinfo);
        }
        AE->set_params(poses, pos_times);
    }
    else if (rdata.type == LOOKAT_DATA)
    {
        float yaw, pitch;
        memcpy(&yaw, rdata.data.c_str(), float_size);
        memcpy(&pitch, rdata.data.c_str() + float_size, float_size);
        SE->set_params(yaw, pitch, motion::HEAD_STATE_LOOKAT);
    }
    else if (rdata.type == JOINT_OFFSET)
    {
        robot_joint_deg jdeg;

        for (size_t i = 0; i < ROBOT->get_joint_map().size(); i++)
        {
            memcpy(&jdeg, rdata.data.c_str() + i * sizeof(robot_joint_deg), sizeof(robot_joint_deg));
            ROBOT->get_joint(jdeg.id)->offset_ = jdeg.deg;
        }
    }
    else if (rdata.type == CAMERA_SET)
    {
        camera_info info;
        memcpy(&(info.id), rdata.data.c_str(), int_size);
        memcpy(&(info.value), rdata.data.c_str() + int_size, float_size);
        VISION->set_camera_info(info);
        shared_ptr<Camera> cm = dynamic_pointer_cast<Camera>(get_sensor("camera"));
        cm->set_camera_info(info);
    }
    else if (rdata.type == IMAGE_SEND_TYPE)
    {
        image_send_type t;
        memcpy(&t, rdata.data.c_str(), enum_size);
        switch (t)
        {
            case IMAGE_SEND_ORIGIN:
            case IMAGE_SEND_RESULT:
                VISION->set_img_send_type(static_cast<image_send_type>(t));
                break;
            case IMAGE_SEND_POINT:
                {
                    int x,y;
                    memcpy(&x, rdata.data.c_str()+enum_size, int_size);
                    memcpy(&y, rdata.data.c_str()+enum_size+int_size, int_size);
                    VISION->get_point_dis(x,y);
                    break;
                }
            case IMAGE_SEND_RECT:
                break;
            default:
                break;
        }
    }

    SERVER->reset_rmt_data();
}

