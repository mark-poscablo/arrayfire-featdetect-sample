cmake_minimum_required(VERSION 3.10)
project (ArrayfireFeatDetectSample)

# Change ArrayFire_DIR to your machine's arrayfire directory
set(ArrayFire_DIR /home/mark/Documents/arrayfire/build/share/ArrayFire/cmake)

set(CMAKE_CXX_STANDARD 11)
# set(CMAKE_CXX_FLAGS "-fsanitize=address -fno-omit-frame-pointer")
set(CMAKE_BUILD_TYPE Debug)
find_package(ArrayFire)
add_executable (feat_detect.out featdetect.cpp)
target_link_libraries(feat_detect.out ArrayFire::af)

