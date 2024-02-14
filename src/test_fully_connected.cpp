#include <gtest/gtest.h>
#include "Eigen/Dense"
#include "FullyConnected.hpp"
#include "Optimizers.hpp"

class FCTest : public testing::Test {
protected:
    const int batchSize = 9;
    const int inputSize = 4;
    const int outputSize = 3;
    Eigen::MatrixXd input_tensor;
    SGD sgd;
    FullyConnected fc;
    void SetUp() override {
        input_tensor = Eigen::MatrixXd::Random(batchSize, inputSize);
        fc = FullyConnected(inputSize, outputSize);
        sgd = SGD(1);
    }

};

TEST_F(FCTest, TestWeightsUpdate) {
    for (int i=0; i<10; i++) {
        auto output_tensor = fc.forward(input_tensor);
        auto error_tensor = -output_tensor;
        fc.backward(error_tensor, sgd);
        auto new_output_tensor = fc.forward(input_tensor);
        ASSERT_TRUE(output_tensor.array().pow(2).sum() < new_output_tensor.array().pow(2).sum());
    }
}

TEST_F(FCTest, TestBackwardSize) {
    auto output_tensor = fc.forward(input_tensor);
    auto error_tensor = fc.backward(output_tensor, sgd);
    ASSERT_EQ(error_tensor.cols(), inputSize);
    ASSERT_EQ(error_tensor.rows(), batchSize);
}

TEST_F(FCTest, TestNumericalBackward) {
    // numerical results tested with python implementation
    input_tensor = Eigen::MatrixXd(4, 4);
    input_tensor << -1, -1, -1, -2, -2,  0,  1,  1, -3, -3, -2, -2, 2, -1, 0, -2;
    Eigen::MatrixXd w(5, 3);
    w << -1, -3, -1, -1,  2, -2, -1,  2, -2, 1, -1,  0, 2,  0,  0;
    fc.setWeights(w);

    auto output = fc.forward(input_tensor);
    auto error = fc.backward(output, sgd); // use output as error
    Eigen::MatrixXd expected_error(5, 3);
    expected_error << -11, -11, -11, 2, -25, 10, 10, -3, -24, -32, -32, 7, 19, -11, -11, 5;

    ASSERT_TRUE(error.isApprox(expected_error));
}

TEST_F(FCTest, TestBiasUpdate) {
    input_tensor = Eigen::MatrixXd::Zero(batchSize, inputSize);
    for (int i=0; i<10; i++) {
        auto output_tensor = fc.forward(input_tensor);
        auto error_tensor = -output_tensor;
        fc.backward(error_tensor, sgd);
        auto new_output_tensor = fc.forward(input_tensor);
        ASSERT_TRUE(output_tensor.array().pow(2).sum() < new_output_tensor.array().pow(2).sum());
    }
}
