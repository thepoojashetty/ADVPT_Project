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
