// Write a test for the relu function

#include <gtest/gtest.h>
#include "Relu.hpp"

TEST(ReluTest, ReluForward) {
    Relu relu;
    Eigen::MatrixXd input(2, 2);
    input << 1, 2,
             3, 4;
    Eigen::MatrixXd output(2, 2);
    output << 1, 2,
              3, 4;
    ASSERT_EQ(output, relu.forward(input));
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