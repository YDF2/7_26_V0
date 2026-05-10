// 缓冲队列
#ifndef __ADAPTER_HPP
#define __ADAPTER_HPP

#include <mutex>
#include <list>
#include <map>
#include <atomic>
#include "robot/robot.hpp"
#include "singleton.hpp"
#include "worldmodel.hpp"

class Adapter: public Singleton<Adapter>
{
public:
    Adapter()
    {
    }

    // 读取各个角度信息
    inline std::map<int, float> get_degs()
    {
        std::map<int, float> degs;
        degs.clear();
        bd_mutex_.lock();

        if (!body_degs_list.empty())
        {
            degs.insert(body_degs_list.front().begin(), body_degs_list.front().end());
            body_degs_list.pop_front();
        }

        bd_mutex_.unlock();
        hd_mutex_.lock();

        if (!head_degs_list.empty())
        {
            degs.insert(head_degs_list.front().begin(), head_degs_list.front().end());
            head_degs_list.pop_front();
        }

        hd_mutex_.unlock();
        if(WM->fall_data()!=FALL_NONE)
        {
            degs[robot::ROBOT->get_joint("jhead1")->jid_]=robot_math::sign(degs[robot::ROBOT->get_joint("jhead1")->jid_])*90.0;
            degs[robot::ROBOT->get_joint("jhead2")->jid_]=0.0;
        }
        return degs;
    }

    inline std::map<int, float> get_body_degs() const
    {
        bd_mutex_.lock();
        std::map<int, float> res = latest_body_deg;
        bd_mutex_.unlock();
        return res;
    }

    inline std::map<int, float> get_head_degs() const
    {
        hd_mutex_.lock();
        std::map<int, float> res = latest_head_deg;
        hd_mutex_.unlock();
        return res;
    }

    inline bool add_body_degs(const std::map<int, float> &jdmap)
    {
        if (!is_alive_)
        {
            return false;
        }

        bd_mutex_.lock();
        body_degs_list.push_back(jdmap);
        latest_body_deg = jdmap;
        bd_mutex_.unlock();
        return true;
    }

    inline bool add_head_degs(std::map<int, float> &jdmap)
    {
        if (!is_alive_)
        {
            return false;
        }
        hd_mutex_.lock();
        head_degs_list.push_back(jdmap);
        latest_head_deg = jdmap;
        hd_mutex_.unlock();
        return true;
    }

    inline bool body_empty() const
    {
        bd_mutex_.lock();
        bool res = body_degs_list.empty();
        bd_mutex_.unlock();
        return res;
    }

    inline bool head_empty() const
    {
        hd_mutex_.lock();
        bool res = head_degs_list.empty();
        hd_mutex_.unlock();
        return res;
    }

    void start()
    {
        robot::joint_ptr j = robot::ROBOT->get_joint("jhead1");
        latest_head_deg[j->jid_] = j->get_deg();
        j = robot::ROBOT->get_joint("jhead2");
        latest_head_deg[j->jid_] = j->get_deg();

        for (auto &j : robot::ROBOT->get_joint_map())
        {
            if (j.second->name_.find("head") == std::string::npos)
            {
                latest_body_deg[j.second->jid_] = j.second->get_deg();
            }
        }

        is_alive_ = true;
    }
    void stop()
    {
        is_alive_ = false;
    }

    std::atomic_bool run_action_;

private:
    std::list< std::map<int, float> > head_degs_list; // 角度list用于存储角度map
    std::list< std::map<int, float> > body_degs_list;
    std::map<int, float> latest_head_deg;
    std::map<int, float> latest_body_deg;
    mutable std::mutex bd_mutex_, hd_mutex_; // 此二互斥锁分别对应上述两个list
    bool is_alive_;
};

#define MADT Adapter::instance()

#endif
