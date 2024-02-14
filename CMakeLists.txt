cmake_minimum_required(VERSION 3.14)

project(AdvPT-mpu VERSION 1.0
              DESCRIPTION "FCN"
              LANGUAGES CXX)

include_directories(
    ${PROJECT_SOURCE_DIR}/include
)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)

FetchContent_MakeAvailable(googletest)


enable_testing()

add_executable(
  test_relu
  src/test_relu.cpp
)
target_link_libraries(
    test_relu
  GTest::gtest_main
)
include(GoogleTest)
gtest_discover_tests(test_relu)

add_executable(
  test_loss
  src/test_loss.cpp
)
target_link_libraries(
  test_loss
  GTest::gtest_main
)
include(GoogleTest)
gtest_discover_tests(test_loss)


add_executable(
  test_softmax
  src/test_softmax.cpp
)
target_link_libraries(
    test_softmax
  GTest::gtest_main
)
include(GoogleTest)
gtest_discover_tests(test_softmax)

add_executable(
  test_fully_connected
  src/test_fully_connected.cpp
)
target_link_libraries(
  test_fully_connected
  GTest::gtest_main
)
include(GoogleTest)
gtest_discover_tests(test_fully_connected)


add_executable(test_data_images src/test_data_images.cpp)
target_compile_features(test_data_images PRIVATE cxx_std_20)
target_compile_options(test_data_images PRIVATE -Wall -Wextra -pedantic -Werror -Wno-unused-parameter -Wno-unused-variable -pg -g)
target_link_options(test_data_images PRIVATE -pg)

add_executable(test_data_labels src/test_data_labels.cpp)
target_compile_features(test_data_labels PRIVATE cxx_std_20)
target_compile_options(test_data_labels PRIVATE -Wall -Wextra -pedantic -Werror -Wno-unused-parameter -Wno-unused-variable -pg -g)
target_link_options(test_data_labels PRIVATE -pg)

add_executable(mnist src/train_test_model.cpp)
target_compile_features(mnist PRIVATE cxx_std_20)
target_compile_options(mnist PRIVATE -Wall -Wextra -pedantic -Werror -Wno-unused-parameter -Wno-unused-variable -pg -g -O3)
target_link_options(mnist PRIVATE -pg)
