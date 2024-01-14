#pragma once
// Write a test for the softmax function

#include <gtest/gtest.h>
#include "Softmax.hpp"
#include "Eigen/Dense"

TEST(SoftmaxTest, SoftmaxForward) {
    Softmax softmax;
    Eigen::MatrixXd input(2, 2);
    input << 1, 2,
             3, 4;
    Eigen::MatrixXd output(2, 2);
    output << 0.0320586, 0.0871443,
              0.2368828, 0.6439143;
    ASSERT_EQ(output, softmax.forward(input));
}

TEST(SoftmaxTest, SoftmaxBackward) {
    Softmax softmax;
    Eigen::MatrixXd input(2, 2);
    input << 1, 2,
             3, 4;
    Eigen::MatrixXd target(2, 2);
    target << 0, 1,
              1, 0;
    Eigen::MatrixXd error(2, 2);
    error << 0.0320586, 0.0871443,
             0.2368828, 0.6439143;
    Eigen::MatrixXd output(2, 2);
    output << -0.0320586, 0.0871443,
              0.2368828, -0.6439143;
    ASSERT_EQ(output, softmax.backward(error, target));
}