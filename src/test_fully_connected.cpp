#include <gtest/gtest.h>
#include "Eigen/Dense"
#include "FullyConnected.hpp"
#include "Optimizers.hpp"

class FCTest : public testing::Test {
protected:
    const int batchSize = 9;
    const int input_size = 4;
    const int output_size = 3;
    Eigen::MatrixXd input_tensor;
    SGD sgd;
    FullyConnected fc;
    void SetUp() override {
        input_tensor = Eigen::MatrixXd::Random(batchSize, input_size);
        fc = FullyConnected(input_size, output_size);
        sgd = SGD(1);
    }

};

TEST_F(FCTest, TestWeightsUpdate) {
    for (int i=0; i<10; i++) {
        Eigen::MatrixXd output_tensor = fc.forward(input_tensor);
        Eigen::MatrixXd error_tensor = -output_tensor;
        fc.backward(error_tensor, sgd);
        Eigen::MatrixXd new_output_tensor = fc.forward(input_tensor);
        ASSERT_TRUE(output_tensor.array().pow(2).sum() < new_output_tensor.array().pow(2).sum());
    }
}
