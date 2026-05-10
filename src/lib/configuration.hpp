#ifndef __CONFIGURATION_HPP
#define __CONFIGURATION_HPP

#include "parser/parser.hpp"
#include "singleton.hpp"
#include <eigen3/Eigen/Dense>

class Configuration: public Singleton<Configuration>
{
public:
    Configuration()
    {
        if (!parser::parse("data/config.conf", config_tree_))
        {
            LOG(LOG_ERROR)<<"Configuration parser error"<<endll;
            exit(1);
        }
        team_number_ = config_tree_.get<int>("team_number");
    }
    
    bool init(const int &id = 0)
    {
        id_ = id;
        player_ = "players." + std::to_string(id_);
        return true;
    }

    std::vector<int> players()
    {
        parser::bpt::ptree pt = config_tree_.get_child("players");
        std::vector<int> res;
        for(auto &t:pt)
        {
            std::string id = t.first;
            res.push_back(std::atoi(id.c_str()));
        }
        return res;
    }
    
    template<typename T>
    inline T get_config_value(const std::string &keyword) const
    {
        try
        {
            return config_tree_.get<T>(keyword);
        }
        catch (parser::bpt::ptree_error &e)
        {
            throw ClassException<Configuration>("No such keyword: " + keyword);
        }
    }

    template<typename T, int size>
    inline Eigen::Matrix<T, size, 1> get_config_vector(const std::string &keyword) const
    {
        Eigen::Matrix<T, size, 1> res = Eigen::Matrix<T, size, 1>::Zero(size, 1);
        int i=0;
        try
        {
            parser::bpt::ptree tpt = config_tree_.get_child(keyword);
            for(auto &t:tpt)
            {
                if(i>=size) break;
                res[i++] = t.second.get_value<T>();
            }
            return res;
        }
        catch (parser::bpt::ptree_error &e)
        {
            throw ClassException<Configuration>("No such keyword: " + keyword);
        }
    }

    std::string get_my_role() const
    {
        parser::bpt::ptree tpt = config_tree_.get_child("strategy");
        for(auto &t:tpt)
        {
            std::string role = t.first;
            parser::bpt::ptree cpt = t.second;
            if(cpt.get<int>("id") == id_)
                return role;
        }
        LOG(LOG_ERROR)<<"can't find my role"<<endll;
        return "";
    }

    inline int id() const
    {
        return id_;
    }
    inline int keeper_id() const
    {
        return config_tree_.get<int>("keeper_id");
    }
    inline int team_number() const
    {
        return team_number_;
    }
    inline std::string player() const
    {
        return player_;
    };
    inline std::string field_file() const
    {
        return config_tree_.get<std::string>("field_file");
    }
    inline std::string robot_file() const
    {
        return config_tree_.get<std::string>(player_ + ".robot_file");
    }
    inline std::string offset_file() const
    {
        return config_tree_.get<std::string>(player_ + ".offset_file");
    }
    inline std::string action_file() const
    {
        return config_tree_.get<std::string>(player_ + ".action_file");
    }

private:
    parser::bpt::ptree config_tree_;
    std::string player_;
    int id_, team_number_;;
};

#define CONF Configuration::instance()

#endif
