add_library(walk STATIC
        walk_engine.cpp
        CubicSpline.cpp
        SmoothSpline.cpp
        HumanoidModel.cpp
        IKWalk.cpp
        LegIK.cpp
        Polynom.cpp
        Spline.cpp)

target_link_libraries(walk pthread)


