cmake_minimum_required(VERSION 3.3.4)
project(track_drone)
project(new_pnp VERSION 0.0.1 DESCRIPTION "new and faster PnP library")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-deprecated-declarations -O3 -march=native ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations -O3 -march=native")

include_directories(${CMAKE_SOURCE_DIR}/src)
find_package(Eigen3 3.3.4 REQUIRED)

# TODO: optional add mkl support for faster and more accurate results
#set(MKLROOT "/home/$USER/intel/compilers_and_libraries_2020.1.217/linux/mkl/")
#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ltbb -D__STDC_CONSTANT_MACROS")
#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread")

find_package( OpenCV 3.2.0 REQUIRED )
set(source_files
        ${PROJECT_SOURCE_DIR}/motion_vectors.cpp
        ${PROJECT_SOURCE_DIR}/Utils.cpp
        ${PROJECT_SOURCE_DIR}/Product.cpp
        ${PROJECT_SOURCE_DIR}/PnPProblem.cpp
        ${PROJECT_SOURCE_DIR}/RobustMatcher.cpp
        ${PROJECT_SOURCE_DIR}/frame.cpp
        ${PROJECT_SOURCE_DIR}/Utils.cpp
        ${PROJECT_SOURCE_DIR}/Projection.cpp
        ${PROJECT_SOURCE_DIR}/PnP/PnpInput.cpp
        ${PROJECT_SOURCE_DIR}/Utils_pnp/Parsing.cpp
        ${PROJECT_SOURCE_DIR}/Utils_pnp/Definitions.h
        ${PROJECT_SOURCE_DIR}/PnP/PnpObjective.cpp
        ${PROJECT_SOURCE_DIR}/PnP/PnpProblemSolver.cpp
        ${PROJECT_SOURCE_DIR}/PnP/DualVar.cpp
        ${PROJECT_SOURCE_DIR}/PnP/PnpBarrierMethodSolver.cpp
        ${PROJECT_SOURCE_DIR}/Utils_pnp/GeneralUtils.cpp
        ${PROJECT_SOURCE_DIR}/PnP/PnpResult.cpp
        ${PROJECT_SOURCE_DIR}/PnP/PnpLocalSearch.cpp
        ${PROJECT_SOURCE_DIR}/PnP/QuaternionVector.cpp
        )
ocv_include_modules_recurse(${OPENCV_CPP_SAMPLES_REQUIRED_DEPS})

# TODO: library creation file and directory
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_library(${PROJECT_NAME} SHARED
        ${source_files}
        )
set_target_properties(${PROJECT_NAME} PROPERTIES VERSION ${PROJECT_VERSION})

# TODO: public headers file
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER
        PnP/PnpProblemSolver.h
#        PnPProblem.h
)
include_directories(
        ${PROJECT_SOURCE_DIR}/PnP
        ${PROJECT_SOURCE_DIR}/Utils_pnp
        ${EIGEN3_INCLUDE_DIR}
        ${OpenCV_INCLUDE_DIRS}
)
include_directories(-Idependencies/include -Ldependencies/lib)

target_include_directories(${PROJECT_NAME} PRIVATE include)
target_include_directories(${PROJECT_NAME} PRIVATE src)

target_link_libraries(${PROJECT_NAME}
        ${OpenCV_LIBS} stdc++fs -ltbb -lstdc++ -lm -ldl -lswscale -lavformat -lavcodec -lswresample -lavutil -lpthread -lz -lc -lrt
        ${EIGEN3_LIBS}
        )

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
add_executable(sample_rotation_library examples/test_simple_rotation.cpp)
target_link_libraries(sample_rotation_library ${PROJECT_NAME})
