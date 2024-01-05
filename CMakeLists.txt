cmake_minimum_required(VERSION 3.16)

project(AdvPT-mpu VERSION 1.0
              DESCRIPTION "FCN"
              LANGUAGES CXX)

include_directories(
    ${PROJECT_SOURCE_DIR}/include
)

#file(GLOB all_SRC
#        "${PROJECT_SOURCE_DIR}/src/*.cpp"
#)

add_executable(test_data_images src/test_data_images.cpp)
target_compile_features(test_data_images PRIVATE cxx_std_20)
target_compile_options(test_data_images PRIVATE -Wall -Wextra -pedantic -Werror -Wno-unused-parameter -Wno-unused-variable -pg -g)
target_link_options(test_data_images PRIVATE -pg)

