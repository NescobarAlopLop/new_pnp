cmake_minimum_required(VERSION 3.3.4)
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
        ${PROJECT_SOURCE_DIR}/PnPProblem.cpp
        ${PROJECT_SOURCE_DIR}/Utils_pnp/Definitions.h
        ${PROJECT_SOURCE_DIR}/Utils_pnp/GeneralUtils.cpp
        ${PROJECT_SOURCE_DIR}/Utils_pnp/Parsing.cpp
        ${PROJECT_SOURCE_DIR}/src/DualVar.cpp
        ${PROJECT_SOURCE_DIR}/src/PnpBarrierMethodSolver.cpp
        ${PROJECT_SOURCE_DIR}/src/PnpInput.cpp
        ${PROJECT_SOURCE_DIR}/src/PnpLocalSearch.cpp
        ${PROJECT_SOURCE_DIR}/src/PnpObjective.cpp
        ${PROJECT_SOURCE_DIR}/src/PnpProblemSolver.cpp
        ${PROJECT_SOURCE_DIR}/src/PnpResult.cpp
        ${PROJECT_SOURCE_DIR}/src/QuaternionVector.cpp
        )

# TODO: library creation file and directory
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_library(${PROJECT_NAME} SHARED
        ${source_files}
        )
set_target_properties(${PROJECT_NAME} PROPERTIES VERSION ${PROJECT_VERSION})

# TODO: public headers file
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER
        include/PnpProblemSolver.h
#        PnPProblem.h
)
include_directories(
        ${PROJECT_SOURCE_DIR}/include
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
add_executable(sample_rotation examples/test_simple_rotation.cpp)
target_link_libraries(sample_rotation ${PROJECT_NAME})
