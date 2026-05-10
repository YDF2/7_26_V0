#ifndef __TASK_HPP
#define __TASK_HPP

#include <memory>
#include <string>
#include <map>
#include <vector>
#include <list>

class Task
{
public:
    Task()=default;
    Task(const std::string &name): name_(name){}

    std::string name() const
    {
        return name_;
    }

    virtual bool perform()
    {
        return true;
    }

private:
    std::string name_;
};

typedef std::shared_ptr<Task> task_ptr;
typedef std::list<task_ptr> task_list;

#endif
