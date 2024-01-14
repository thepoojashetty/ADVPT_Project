#pragma once
// Write a test for the relu function

#include <gtest/gtest.h>
// import relu.hpp
#include "Relu.hpp"
#include "Eigen/Dense"

TEST(ReluTest, ReluForward) {
    // WIP
    Relu relu;
    Eigen::MatrixXd input(2, 2);
    input << -1, 2,
             3, -4;
    // apply relu.forward(input)
    Eigen::MatrixXd testRes(2, 2);
    testRes = relu.forward(input);
    std::cout << testRes << std::endl;
    // check if the result is correct
    Eigen::MatrixXd output(2, 2);
    output << 0, 0,
              0, 0;
    ASSERT_EQ(output, testRes);
}

TEST(ReluTest, ReluBackward) {
    Relu relu;
    Eigen::MatrixXd input(2, 2);
    input << 1, 2,
             3, 4;
    Eigen::MatrixXd error(2, 2);
    error << 1, 2,
             3, 4;
    Eigen::MatrixXd output(2, 2);
    output << 1, 2,
              3, 4;
    ASSERT_EQ(output, relu.backward(error));
}
