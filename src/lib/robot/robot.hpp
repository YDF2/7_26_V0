// 当前机器人数据模型，每次从缓冲区取出数据存到此处，再发给机器人
#ifndef __ROBOT_HPP
#define __ROBOT_HPP

#include <string>
#include <mutex>
#include "robot_define.hpp"
#include "singleton.hpp"
#include "math/math.hpp"

namespace robot
{

    class Robot: public Singleton<Robot>
    {
    public:
        robot_math::TransformMatrix get_foot_mat_from_pose(const robot_pose &pose, bool left);
        robot_math::TransformMatrix get_body_mat_from_pose(const robot_pose &pose);

        bool arm_inverse_kinematics(const Eigen::Vector3d &hand, std::vector<double> &deg);

        robot_math::TransformMatrix leg_forward_kinematics(std::vector<double> degs, bool left);
        bool leg_inverse_kinematics(const robot_math::TransformMatrix &body, const robot_math::TransformMatrix &foot,
                                    std::vector<double> &deg, const bool &left = false);
        bool leg_inverse_kinematics_walk(const robot_math::TransformMatrix &body, const robot_math::TransformMatrix &foot,
                                    std::vector<double> &deg, const bool &left = false);

        void init(const std::string &robot_file, const std::string &action_file, const std::string &offset_file);

        void set_degs(const std::map<int, float> &jdmap);
        void set_real_degs(const std::map<int, float> &jdmap);

        std::vector<double> get_foot_degs(int support);
        std::vector<double> get_head_degs()
        {
            //std::lock_guard<std::mutex> lk(robot_mtx_);
            std::vector<double> res = {joint_map_["jhead1"]->get_deg(), joint_map_["jhead2"]->get_deg()};
            return res;
        }
        
        joint_ptr get_real_joint(const int &id);                 //for real joints
        joint_ptr get_real_joint(const std::string &name);       //for real joints

        joint_ptr get_joint(const int &id);
        joint_ptr get_joint(const std::string &name);

        double trunk_length()
        {
            return bone_map_["torso"]->length_;
        }

        double neck_length()
        {
            return bone_map_["head1"]->length_;
        }

        double head_length()
        {
            return bone_map_["camera1"]->length_;
        }

        double A() const
        {
            return A_;
        }
        double B() const
        {
            return B_;
        }
        double C() const
        {
            return C_;
        }
        double D() const
        {
            return D_;
        }
        double E() const
        {
            return E_;
        }
        double F() const
        {
            return F_;
        }
        double leg_length() const
        {
            return A_ + B_ + C_;
        }
        double leg_length_without_foot() const
        {
            return A_ + B_;
        }

        joint_map get_joint_map()
        {
            std::lock_guard<std::mutex> lk(robot_mtx_);
            return joint_map_;
        }
        bone_map &get_bone_map()
        {
            return bone_map_;
        }
        bone_map get_bone_map() const
        {
            return bone_map_;
        }
        act_map &get_act_map()
        {
            return act_map_;
        }
        act_map get_act_map() const
        {
            return act_map_;
        }
        pos_map &get_pos_map()
        {
            return pos_map_;
        }
        pos_map get_pos_map() const
        {
            return pos_map_;
        }
        bone_ptr get_main_bone()
        {
            return main_bone_;
        }

        joint_map &get_realJoint_map()
        {
            return real_joint_map_;
        }
        joint_map get_realJoint_map() const
        {
            return real_joint_map_;
        }

    private:
        mutable std::mutex robot_mtx_;
        bone_ptr main_bone_;
        joint_map joint_map_;
        joint_map real_joint_map_;

        bone_map bone_map_;
        act_map act_map_;
        pos_map pos_map_;

        double D_, A_, B_, C_;
        double E_, F_;
    };

#define ROBOT Robot::instance()
}

#endif
