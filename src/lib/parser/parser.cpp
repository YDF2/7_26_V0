#include "parser.hpp"

namespace parser
{
#define acts_key_ "acts"
#define poses_key_ "poses"

    extern robot::joint_ptr parse_joint(const std::string &j_name, const bpt::ptree &pt, robot::bone_map &bone_map, robot::joint_map &joint_map);
    extern robot::bone_ptr parse_bone(const bpt::ptree &pt, robot::bone_map &bone_map, robot::joint_map &joint_map);
    extern bool parse_ptree(bpt::ptree &pt);
    
    bool get_tree_from_file(const std::string &filename, bpt::ptree &pt)
    {
        std::ifstream ifs(filename.c_str(), std::ios::in);
        std::string line;
        int count_of_quotatuion = 0;
        std::stringstream json_data;

        if (!ifs)
        {
            return false;
        }

        while (std::getline(ifs, line))
        {
            count_of_quotatuion = 0;

            for (int i = 0; i < line.size(); i++)
            {
                if (line[i] == '\'' || line[i] == '\"')
                {
                    count_of_quotatuion++;
                }

                if (i < line.size() - 2)
                {
                    if (line[i] == '/' && line[i + 1] == '/' && count_of_quotatuion % 2 == 0)
                    {
                        break;
                    }
                }

                json_data << line[i];
            }

            line.clear();
        }

        bpt::read_json(json_data, pt);
        return true;
    }

    void write_tree_to_file(const std::string &filename, const bpt::ptree &pt)
    {
        std::ostringstream os;
        bpt::write_json(os, pt);
        std::ofstream tree(filename.c_str());
        tree << os.str();
        tree.close();
    }

    robot::bone_ptr parse_bone(const bpt::ptree &pt, robot::bone_map &bone_map, robot::joint_map &joint_map)
    {
        robot::bone_ptr b = std::make_shared<robot::Bone>();
        b->name_ = pt.get<std::string>("name");
        //std::cout<<b->name_<<"\n";
        b->length_ = pt.get<float>("length");
        bpt::ptree c_pt = pt.get_child("cp");
        std::vector<float> temp;
        temp.clear();

        for (auto &p : c_pt)
        {
            temp.push_back(p.second.get_value<float>());
        }

        b->cp_ << temp[0], temp[1], temp[2];

        c_pt = pt.get_child("cr");
        temp.clear();

        for (auto &r : c_pt)
        {
            temp.push_back(r.second.get_value<float>());
        }

        b->cr_ << temp[0], temp[1], temp[2];
        b->joints_.clear();
        bone_map[b->name_] = b;

        try
        {
            bpt::ptree cb = pt.get_child("joints");

            for (auto &j : cb)
            {
                b->joints_.push_back(parse_joint(j.first, j.second, bone_map, joint_map));
            }
        }
        catch (bpt::ptree_error &e)
        {
            return b;
        }

        return b;
    }

    robot::joint_ptr parse_joint(const std::string &j_name, const bpt::ptree &pt, robot::bone_map &bone_map, robot::joint_map &joint_map)
    {
        robot::joint_ptr j = std::make_shared<robot::Joint>();
        robot::bone_ptr b = std::make_shared<robot::Bone>();
        bpt::ptree cb;
        //std::cout<<j_name<<"\n";
        j->name_ = j_name;
        j->can_turn_ = pt.get<bool>("ct");

        bpt::ptree c_pt = pt.get_child("cp");
        std::vector<float> temp;
        temp.clear();

        for (auto &p : c_pt)
        {
            temp.push_back(p.second.get_value<float>());
        }

        j->cp_ << temp[0], temp[1], temp[2];
        c_pt = pt.get_child("cr");
        temp.clear();

        for (auto &r : c_pt)
        {
            temp.push_back(r.second.get_value<float>());
        }

        j->cr_ << temp[0], temp[1], temp[2];

        if (j->can_turn_)
        {
            j->init_deg_ = pt.get<float>("init");
            j->jid_ = pt.get<int>("jid");
            j->set_deg(pt.get<float>("cur"));

            try
            {
                j->inverse_ = (pt.get<bool>("inver") ? -1.0 : 1.0);
            }
            catch (bpt::ptree_error &e)
            {
                j->inverse_ = 1.0;
            }

            joint_map[j_name] = j;
        }

        try
        {
            j->next_bone_ = parse_bone(pt.get_child("bone"), bone_map, joint_map);
        }
        catch (bpt::ptree_error &e)
        {
            LOG(LOG_ERROR) << e.what() << endll;
        }

        return j;
    }

    robot::bone_ptr parse(const std::string &filename, robot::bone_map &bone_map, robot::joint_map &joint_map)
    {
        bpt::ptree pt;
        if (!get_tree_from_file(filename, pt))
            return nullptr;
        return parse_bone(pt, bone_map, joint_map);
    }

    robot::robot_pose get_pose_from_string(const std::string &str)
    {
        std::stringstream ss;
        ss<<str;
        robot::robot_pose pose;
        ss>>pose.x;
        ss>>pose.y;
        ss>>pose.z;
        ss>>pose.pitch;
        ss>>pose.roll;
        ss>>pose.yaw;
        return pose;
    }

    std::string get_string_from_pose(const robot::robot_pose &pose)
    {
        std::string str="";
        str += (std::to_string(pose.x)+" ");
        str += (std::to_string(pose.y)+" ");
        str += (std::to_string(pose.z)+" ");
        str += (std::to_string(pose.pitch)+" ");
        str += (std::to_string(pose.roll)+" ");
        str += (std::to_string(pose.yaw));;
        return str;
    }

    bool pos_exist(const std::string &name, const std::map<std::string, robot::robot_pos> &poses)
    {
        if (poses.find(name) != poses.end())
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    bool parse(const std::string &act_file, std::map<std::string, robot::robot_act> &acts, std::map<std::string, robot::robot_pos> &poses)
    {
        bpt::ptree pt;
        acts.clear();
        poses.clear();

        if (!get_tree_from_file(act_file, pt))
            return false;

        bpt::ptree act_pt, pos_pt;

        try
        {
            act_pt = pt.get_child(acts_key_);
            pos_pt = pt.get_child(poses_key_);
        }
        catch (bpt::ptree_error &e)
        {
            return false;
        }

        robot::robot_pos t_pos;

        for (auto &pos : pos_pt)
        {
            t_pos.name.clear();
            t_pos.pose_info.clear();
            t_pos.name = pos.first;

            for (auto &info : pos.second)
            {
                t_pos.pose_info[robot::get_motion_by_name(info.first)] = get_pose_from_string(info.second.data());
            }

            poses[t_pos.name] = t_pos;
        }

        robot::robot_act t_act;
        robot::robot_one_pos t_one_pos;

        for (auto &act : act_pt)
        {
            t_act.name.clear();
            t_act.poses.clear();
            t_act.name = act.first;

            for (auto &pos : act.second)
            {
                if (pos_exist(pos.first, poses))
                {
                    t_one_pos.act_time = 0;
                    t_one_pos.pos_name.clear();
                    t_one_pos.pos_name = pos.first;
                    t_one_pos.act_time = pos.second.get_value<int>();
                    t_act.poses.push_back(t_one_pos);
                }
                else
                {
                    LOG(LOG_ERROR)<<"pos: " + pos.first + " does not exist"<<endll;
                    return false;
                }
            }

            acts[t_act.name] = t_act;
        }
        return true;
    }

    void save(const std::string &act_file, const std::map<std::string, robot::robot_act> &acts, const std::map<std::string, robot::robot_pos> &poses)
    {
        bpt::ptree pt;
        bpt::ptree act_pt, pos_pt;
        
        std::unordered_set<std::string> saved_poses;

        bpt::ptree act_info_child;
        bpt::ptree pos_info_child;
        
        for (auto &act : acts)
        {
            act_info_child.clear();

            for (size_t i = 0; i < act.second.poses.size(); i++)
            {
                std::string pos_name = act.second.poses[i].pos_name;
                act_info_child.add<int>(pos_name, act.second.poses[i].act_time);
                if(saved_poses.find(pos_name)!=saved_poses.end()) 
                    continue;
                pos_info_child.clear();
                auto pos_iter = poses.find(pos_name);
                for (auto &p_info : pos_iter->second.pose_info)
                {
                    pos_info_child.add(robot::get_name_by_motion(p_info.first), get_string_from_pose(p_info.second));
                }
                pos_pt.add_child(pos_name, pos_info_child);
                saved_poses.insert(pos_name);
            }

            act_pt.add_child(act.second.name, act_info_child);
        }

        pt.add_child(acts_key_, act_pt);
        pt.add_child(poses_key_, pos_pt);
        write_tree_to_file(act_file, pt);
    }

    bool parse(const std::string &filename, std::map<std::string, camera_info> &ctrl_infos)
    {
        bpt::ptree pt;

        ctrl_infos.clear();

        if (!get_tree_from_file(filename, pt))
        {
            return false;
        }

        for (auto &para : pt)
        {
            camera_info info;
            info.name = para.first;
            info.id = para.second.get<int>("id");
            info.value = para.second.get<float>("value");
            info.default_value = para.second.get<float>("default");
            info.min_value = para.second.get<float>("min");
            info.max_value = para.second.get<float>("max");
            ctrl_infos[info.name] = info;
        }
        return true;
    }

    void save(const std::string &filename, const std::map<std::string, camera_info> &ctrl_infos)
    {
        bpt::ptree pt;

        for (auto item : ctrl_infos)
        {
            bpt::ptree item_pt;
            item_pt.add("id", item.second.id);
            item_pt.add("value", item.second.value);
            item_pt.add("default", item.second.default_value);
            item_pt.add("min", item.second.min_value);
            item_pt.add("max", item.second.max_value);
            pt.add_child(item.first, item_pt);
        }

        write_tree_to_file(filename, pt);
    }

    bool parse(const std::string &filename, camera_param &param)
    {
        bpt::ptree pt;

        if (!get_tree_from_file(filename, pt))
        {
            return false;
        }
        param.fx = pt.get<float>("fx");
        param.fy = pt.get<float>("fy");
        param.cx = pt.get<float>("cx");
        param.cy = pt.get<float>("cy");
        param.k1 = pt.get<float>("k1");
        param.k2 = pt.get<float>("k2");
        param.p1 = pt.get<float>("p1");
        param.p2 = pt.get<float>("p2");

        return true;
    }

    bool add_child(bpt::ptree &oript, const std::string &key, const bpt::ptree &pt)
    {
        try
        {
            oript.erase(key);
            oript.add_child(key, pt);
        }
        catch (bpt::ptree_error &e)
        {
            return false;
        }

        return true;
    }

    bool parse_file(const std::string &cfgname, bpt::ptree &pt)
    {
        if (get_tree_from_file(cfgname, pt))
        {
            return parse_ptree(pt);
        }
        else
        {
            return false;
        }
    }

    bool parse_ptree(bpt::ptree &pt)
    {
        std::string data;
        bpt::ptree tpt;
        bpt::ptree oript = pt;
        int size = oript.size();

        if (size == 1)
        {
            data.clear();
            data = oript.begin()->second.data();

            if (data.size() == 0)
            {
                tpt = oript.begin()->second;

                if (parse_ptree(tpt))
                {
                    if (!add_child(pt, oript.begin()->first, tpt))
                    {
                        return false;
                    }
                    else
                    {
                        return true;
                    }
                }
                else
                {
                    return false;
                }
            }
            else
            {
                if (oript.begin()->first.empty())
                {
                    if (parse_file(data, tpt))
                    {
                        pt = tpt;
                    }
                    else
                    {
                        return false;
                    }

                    return true;
                }
            }
        }

        auto iter = oript.begin();

        while (iter != oript.end())
        {
            tpt = iter->second;
            data.clear();
            data = tpt.data();

            if (data.size() == 0)
            {
                if (parse_ptree(tpt))
                {
                    if (!add_child(pt, iter->first, tpt))
                    {
                        return false;
                    }
                }
                else
                {
                    return false;
                }
            }

            iter++;
        }

        return true;
    }

    bool parse(const std::string &cfgname, bpt::ptree &pt)
    {
        return parse_file(cfgname, pt);
    }

    bool parse(const std::string &filename, filed_info &field)
    {
        bpt::ptree pt;

        if (!get_tree_from_file(filename, pt))
        {
            return false;
        }

        field.field_length = pt.get<int>("field_length");
        field.field_width = pt.get<int>("field_width");
        field.goal_depth = pt.get<int>("goal_depth");
        field.goal_width = pt.get<int>("goal_width");
        field.goal_height = pt.get<int>("goal_height");
        field.goal_area_length = pt.get<int>("goal_area_length");
        field.goal_area_width = pt.get<int>("goal_area_width");
        field.penalty_mark_distance = pt.get<int>("penalty_mark_distance");
        field.center_circle_diameter = pt.get<int>("center_circle_diameter");
        field.border_strip_width_min = pt.get<int>("border_strip_width_min");
        return true;
    }

    bool parse(const std::string &filename, robot::joint_map &jmap)
    {
        bpt::ptree pt;

        if (!get_tree_from_file(filename, pt))
        {
            return false;
        }

        for (auto &offset : pt)
        {
            auto it = jmap.find(offset.first);
            if (it == jmap.end() || !it->second)
            {
                LOG(LOG_WARN) << "skip unknown joint in offset file: " << offset.first << endll;
                continue;
            }

            try
            {
                it->second->offset_ = offset.second.get_value<float>();
            }
            catch (bpt::ptree_error &e)
            {
                LOG(LOG_ERROR) << "invalid offset value for joint: " << offset.first << ", " << e.what() << endll;
                return false;
            }
        }

        return true;
    }

    void save(const std::string &filename, const robot::joint_map &jmap)
    {
        bpt::ptree pt;

        for (auto &j : jmap)
        {
            pt.add(j.first, j.second->offset_);
        }

        write_tree_to_file(filename, pt);
    }
};