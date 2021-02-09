cmake_minimum_required(VERSION 2.8.4)
project(track_drone)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(sample_dir ${CMAKE_CURRENT_SOURCE_DIR}/)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-deprecated-declarations -O3 -march=native ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations -O3 -march=native")

include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/eigen-3.3.7)

#set(MKLROOT "/home/firas/intel/compilers_and_libraries_2020.1.217/linux/mkl/")
#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ltbb -D__STDC_CONSTANT_MACROS")
#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}   -std=c++11 -pthread")
#set(target firas_)
find_package( OpenCV 3.2.0 REQUIRED )
set(sample_pnplib
        ${sample_dir}motion_vectors.cpp
        ${sample_dir}Utils.cpp
        ${sample_dir}Product.cpp
        ${sample_dir}PnPProblem.cpp
        ${sample_dir}RobustMatcher.cpp
        ${sample_dir}frame.cpp
        ${sample_dir}Utils.cpp
        ${sample_dir}Projection.cpp
        ${sample_dir}PnP/PnpInput.cpp
        ${sample_dir}Utils_pnp/Parsing.cpp
        ${PROJECT_SOURCE_DIR}/Utils_pnp/Definitions.h
        ${sample_dir}PnP/PnpObjective.cpp
        ${sample_dir}PnP/PnpProblemSolver.cpp
        ${sample_dir}PnP/DualVar.cpp
        ${sample_dir}PnP/PnpBarrierMethodSolver.cpp
        ${sample_dir}Utils_pnp/GeneralUtils.cpp
        ${sample_dir}PnP/PnpResult.cpp
        ${sample_dir}PnP/PnpLocalSearch.cpp
        ${sample_dir}PnP/QuaternionVector.cpp
        )
ocv_include_modules_recurse(${OPENCV_CPP_SAMPLES_REQUIRED_DEPS})

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(Utils_pnp PnP)
include_directories(eigen-3.3.7)
include_directories(-Idependencies/include -Ldependencies/lib)

add_executable(track_drone examples/main.cpp ${sample_pnplib})
target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} stdc++fs -ltbb -lstdc++ -lpthread -lm -ldl
        -lswscale -lavformat -lavcodec -lswresample -lavutil -lpthread -lz -lc -lrt)
target_link_libraries( track_drone ${OpenCV_LIBS} stdc++fs -ltbb -lstdc++ -lm -ldl -lswscale -lavformat -lavcodec -lswresample -lavutil -lpthread -lz -lc -lrt)


add_executable(sample_rotation examples/test_simple_rotation.cpp ${sample_pnplib})
target_link_libraries(sample_rotation ${OpenCV_LIBS} stdc++fs -ltbb -lstdc++ -lm -ldl -lswscale -lavformat -lavcodec -lswresample -lavutil -lpthread -lz -lc -lrt)

add_executable(opencv_units examples/opencv_units.cpp)
target_link_libraries(opencv_units ${OpenCV_LIBS})
