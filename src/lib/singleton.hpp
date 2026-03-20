#ifndef __SINGLETON_HPP
#define __SINGLETON_HPP

#include <memory>

template<class T>
class Singleton
{
public:
    static std::shared_ptr<T> instance()
    {
        if (instance_ == nullptr)
        {
            instance_ = std::make_shared<T>();
        }

        return instance_;
    }
protected:
    Singleton() {}
    T &operator=(const T &) {}
    static std::shared_ptr<T> instance_;
};

template<class T> std::shared_ptr<T> Singleton<T>::instance_ = nullptr;

#endif
