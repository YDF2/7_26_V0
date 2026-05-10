#include "robot.hpp"
#include "parser/parser.hpp"

namespace robot
{
    using namespace std;
    using namespace Eigen;
    using namespace robot_math;

    TransformMatrix Robot::get_foot_mat_from_pose(const robot_pose &pose, bool left)
    {
        double sg = (left ? 1.0 : -1.0);
        TransformMatrix foot_mat;
        Quaternion<double> quat;
        AngleAxisd yawRot, pitchRot, rollRot;

        yawRot = AngleAxisd(deg2rad(pose.yaw), Vector3d::UnitZ());
        pitchRot = AngleAxisd(deg2rad(pose.pitch), Vector3d::UnitY());
        rollRot = AngleAxisd(deg2rad(pose.roll), Vector3d::UnitX());
        quat = rollRot * pitchRot * yawRot;
        foot_mat.set_p(Vector3d(pose.x, pose.y + sg * D_ / 2.0, pose.z));
        foot_mat.set_R(quat.matrix());
        return foot_mat;
    }

    TransformMatrix Robot::get_body_mat_from_pose(const robot_pose &pose)
    {
        TransformMatrix body_mat;
        Quaternion<double> quat;
        AngleAxisd yawRot, pitchRot, rollRot;

        yawRot = AngleAxisd(deg2rad(pose.yaw), Vector3d::UnitZ());
        pitchRot = AngleAxisd(deg2rad(pose.pitch), Vector3d::UnitY());
        rollRot = AngleAxisd(deg2rad(pose.roll), Vector3d::UnitX());
        quat = rollRot * pitchRot * yawRot;
        body_mat.set_p(Vector3d(pose.x, pose.y, pose.z + leg_length()));
        body_mat.set_R(quat.matrix());
        return body_mat;
    }

    bool Robot::arm_inverse_kinematics(const Vector3d &hand, vector<double> &deg)
    {
        double x = hand[0];
        double z = hand[2] - (E_ + F_);
        double l = sqrt(x * x + z * z);

        if (l > E_ + F_)
        {
            return false;
        }

        double q3t = acos((E_ * E_ + F_ * F_ - l * l) / (2 * E_ * F_));
        double q3 = M_PI - q3t;
        double q1t0 = atan2(z, x);
        double q1t1;

        if (z <= 0)
        {
            q1t1 = -M_PI / 2.0 - q1t0;
        }
        else
        {
            if (x <= 0)
            {
                q1t1 = 3.0 * M_PI / 2.0 - q1t0;
            }
            else
            {
                q1t1 = -(M_PI / 2.0 + q1t0);
            }
        }

        double q1t2 = acos((E_ * E_ - F_ * F_ + l * l) / (2 * E_ * l));
        double q1 = (q1t1 + q1t2);
        deg.clear();
        deg.push_back(q1);
        deg.push_back(0.0);
        deg.push_back(q3);
        return true;
    }

    TransformMatrix Robot::leg_forward_kinematics(vector<double> degs, bool left)
    {
        double sg = (left ? 1.0 : -1.0);

        if (degs.size() < 6)
        {
            return TransformMatrix();
        }
        TransformMatrix foot;
        TransformMatrix body;
        body = foot*TransformMatrix(0, 0, C_)*TransformMatrix(-degs[5], 'x')*TransformMatrix(-degs[4], 'y')
                *TransformMatrix(0, 0, B_)*TransformMatrix(-degs[3], 'y')*TransformMatrix(0, 0, A_)
                *TransformMatrix(-degs[2], 'y')*TransformMatrix(-degs[1], 'x')*TransformMatrix(-degs[0], 'z')
                *TransformMatrix(0, -sg*D_ / 2.0, bone_map_["rhip2"]->length_);
        return body;
    }

    bool Robot::leg_inverse_kinematics(const TransformMatrix &body,
                                          const TransformMatrix &foot,
                                          vector<double> &deg, const bool &left)
    {
        double sg = (left ? 1.0 : -1.0);
        Vector3d p16 = foot.p() + C_ * foot.a();
        Vector3d p11 = body.p() + body.R() * Vector3d(0, sg * D_ / 2.0, 0);
        Vector3d r = foot.R().transpose() * (p11 - p16);
        double Lr = r.norm();

        if (Lr > A_ + B_)
        {
            return false;
        }

        double alpha = acos((A_ * A_ + B_ * B_ - Lr * Lr) / (2 * A_ * B_));
        double q4 = M_PI - alpha;

        double beta = asin(A_ * sin(alpha) / Lr);
        double q5 = -atan2(r[0], sign(r[2]) * sqrt(r[1] * r[1] + r[2] * r[2])) - beta;

        double q6 = atan2(r[1], r[2]);

        if (q6 > M_PI / 2.0)
        {
            q6 = q6 - M_PI;
        }
        else if (q6 < -M_PI / 2.0)
        {
            q6 = q6 + M_PI;
        }

        MatrixX3d R = body.R().transpose() * foot.R() * RotX(-q6) * RotY(-q5) * RotY(-q4);
        double q1 = atan2(-R(0, 1), R(1, 1));
        double q3 = atan2(-R(2, 0), R(2, 2));
        double cz = cos(q1), sz = sin(q1);
        double q2 = atan2(R(2, 1), -R(0, 1) * sz + R(1, 1) * cz);

        deg.clear();
        deg.push_back(q1);
        deg.push_back(q2);
        deg.push_back(q3);
        deg.push_back(q4);
        deg.push_back(q5);
        deg.push_back(q6);
        return true;
    }

    bool Robot::leg_inverse_kinematics_walk(const TransformMatrix &body, const TransformMatrix &foot,
                                          vector<double> &deg, const bool &left)
    {   
        double sg = (left ? 1.0 : -1.0);
        
        Vector3d p16 = foot.p();

        Vector3d p11 = body.p() + body.R() * Vector3d(0, sg * D_ / 2.0, 0);
        Vector3d r = foot.R().transpose() * (p11 - p16);

        double Lr = r.norm();

        if (Lr > A_ + B_)
        {   
            //cout<<"inverse_kinematics  "<<"Lr>A+B"<<endl;
            return false;
        }

        double jkeen = M_PI - acos((pow(A_, 2) + pow(B_, 2) - pow(Lr , 2)) / (2 * A_ * B_)); 

        double jankle1 = atan2(r[1], r[2]);
      
        if (jankle1 > M_PI / 2.0)
        {
            jankle1 = jankle1 - M_PI;
        }
        else if (jankle1 < -M_PI / 2.0)
        {
            jankle1 = jankle1 + M_PI;
        } 

        double beta =  asin(A_ * sin(jkeen) / Lr);
        double jankle2 = -atan2( r[0], sign( r[2] )* sqrt(pow(r[1], 2) + pow(r[2], 2)) )- beta;  //踝关节pitch

        

        MatrixX3d R = body.R().transpose() * foot.R() * RotX(-jankle1) * RotY(-jankle2) * RotY(-jkeen);
        double jhip3 = atan2(-R(0, 1), R(1, 1));
        double jhip1 = atan2(-R(2, 0), R(2, 2));
        double cz = cos(jhip3), sz = sin(jhip3);
        double jhip2 = atan2(R(2, 1), -R(0, 1) * sz + R(1, 1) * cz);  

        deg.clear();
        deg.push_back(jhip3);     //jhip3  q1
        deg.push_back(jhip2);     //jhip2  q2
        deg.push_back(jhip1);     //jhip1  q3
        deg.push_back(jkeen);     //jkeen    q4
        deg.push_back(jankle2);   //jankle2  q5
        deg.push_back(jankle1);   //jankle1  q6
        return true;
  
    }

    void Robot::init(const std::string &robot_file, const std::string &action_file,
                        const std::string &offset_file)
    {
        main_bone_ = parser::parse(robot_file, bone_map_, joint_map_);
        if (!main_bone_)
        {
            throw ClassException<Robot>("failed to parse robot file: " + robot_file);
        }

        real_joint_map_ = joint_map_;              //for real joints

        if (!parser::parse(action_file, act_map_, pos_map_))
        {
            throw ClassException<Robot>("failed to parse action file: " + action_file);
        }
        if (!parser::parse(offset_file, joint_map_))
        {
            throw ClassException<Robot>("failed to parse offset file: " + offset_file);
        }

        auto require_bone = [this](const std::string &name) -> bone_ptr {
            auto it = bone_map_.find(name);
            if (it == bone_map_.end() || !it->second)
            {
                throw ClassException<Robot>("missing bone in robot model: " + name);
            }
            return it->second;
        };

        D_ = require_bone("hip")->length_;
        A_ = require_bone("rthigh")->length_;
        B_ = require_bone("rshank")->length_;
        C_ = require_bone("rfoot1")->length_;
        E_ = require_bone("ruparm")->length_;
        F_ = require_bone("rlowarm")->length_;
    }

    void Robot::set_real_degs(const std::map<int, float> &jdmap)
    {
        for (auto &j : jdmap)
        {
            get_real_joint(j.first)->set_deg(j.second);
        }
    }

    void Robot::set_degs(const std::map<int, float> &jdmap)
    {
        std::lock_guard<std::mutex> lk(robot_mtx_);
        for (auto &j : jdmap)
        {
            get_joint(j.first)->set_deg(j.second);
        }
    }

    std::vector<double> Robot::get_foot_degs(int support)
    {
        std::lock_guard<std::mutex> lk(robot_mtx_);
        std::vector<double> res;
        if(support == LEFT_SUPPORT)
        {
            res.push_back(joint_map_["jlhip3"]->get_deg());
            res.push_back(joint_map_["jlhip2"]->get_deg());
            res.push_back(joint_map_["jlhip1"]->get_deg());
            res.push_back(joint_map_["jlknee"]->get_deg());
            res.push_back(joint_map_["jlankle2"]->get_deg());
            res.push_back(joint_map_["jlankle1"]->get_deg());
        }
        else
        {
            res.push_back(joint_map_["jrhip3"]->get_deg());
            res.push_back(joint_map_["jrhip2"]->get_deg());
            res.push_back(joint_map_["jrhip1"]->get_deg());
            res.push_back(joint_map_["jrknee"]->get_deg());
            res.push_back(joint_map_["jrankle2"]->get_deg());
            res.push_back(joint_map_["jrankle1"]->get_deg());
        }
        return res;
    }

    joint_ptr Robot::get_joint(const int &id)
    {
        for (auto &j : joint_map_)
            if (j.second->jid_ == id)
            {
                return j.second;
            }

        throw ClassException<Robot>("cannot find joint by id: " + std::to_string(id));
    }

    joint_ptr Robot::get_joint(const std::string &name)
    {
        for (auto &j : joint_map_)
            if (j.second->name_ == name)
            {
                return j.second;
            }

        throw ClassException<Robot>("cannot find joint by name: " + name);
    }

    joint_ptr Robot::get_real_joint(const int &id)
    {
        for (auto &j : real_joint_map_)
            if (j.second->jid_ == id)
            {
                return j.second;
            }

        throw ClassException<Robot>("cannot find real joint by id: " + std::to_string(id));
    }

    joint_ptr Robot::get_real_joint(const std::string &name)
    {
        for (auto &j : real_joint_map_)
            if (j.second->name_ == name)
            {
                return j.second;
            }

        throw ClassException<Robot>("cannot find real joint by name: " + name);
    }
}