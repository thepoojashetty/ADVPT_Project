#include <gtest/gtest.h>
#include "Eigen/Dense"
#include "Loss.hpp"
#include "Softmax.hpp"

class LossTest: public testing::Test {
protected:
    CrossEntropyLoss loss;
    Eigen::MatrixXd inputTensor;
    Eigen::MatrixXd labelTensor;

    void SetUp() override {
        loss = CrossEntropyLoss();
        labelTensor = Eigen::MatrixXd(3, 3);
        labelTensor << 0, 1, 0, 1, 0, 0, 0, 1, 0;
        inputTensor = Eigen::MatrixXd(3, 3);
        inputTensor << 1.0, 2.0, 3.0,
                   4.0, 5.0, 6.0,
                   7.0, 8.0, 9.0;
    }
};

TEST_F(LossTest, TestForward) {
    
    double output = loss.forward(inputTensor, labelTensor);
    ASSERT_DOUBLE_EQ(output, -4.1588830833596715);
}

TEST_F(LossTest, TestBackward) {
    Eigen::MatrixXd expected_out(3, 3);
    expected_out << 0., -0.5, 0., -0.25, 0., 0., 0., -0.125, 0.;

    Eigen::MatrixXd output = loss.backward(labelTensor);

    ASSERT_TRUE(output.isApprox(expected_out));
}
