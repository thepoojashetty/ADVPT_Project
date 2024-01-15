#include <gtest/gtest.h>
#include "Eigen/Dense"
#include "Relu.hpp"  // Include the header file where ReLULayer is defined

class ReLULayerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize the ReLULayer for each test
        reluLayer = Relu();
    }

    // Declare the ReLULayer instance
    Relu reluLayer;
};

TEST_F(ReLULayerTest, ForwardPass) {
    // Example input values
    Eigen::MatrixXd inputTensor(3, 3);
    inputTensor << -1, 2, -3,
                   4, -5, 6,
                   -7, 8, 9;

    // Forward pass
    Eigen::MatrixXd outputTensor = reluLayer.forward(inputTensor);

    // Check if the output tensor has non-negative values
    ASSERT_TRUE((outputTensor.array() >= 0).all());
}

TEST_F(ReLULayerTest, BackwardPass) {
    // Example input values
    Eigen::MatrixXd inputTensor(3, 3);
    inputTensor << -1, 2, -3,
                   4, -5, 6,
                   -7, 8, 9;

    // Forward pass (required for the backward pass)
    reluLayer.forward(inputTensor);

    // Example error tensor for the backward pass
    Eigen::MatrixXd errorTensor(3, 3);
    errorTensor << 0.1, -0.2, 0.3,
                   -0.4, 0.5, -0.6,
                   0.7, -0.8, 0.9;

    // Backward pass
    Eigen::MatrixXd gradientInput = reluLayer.backward(errorTensor);

    // Check if the gradient with respect to input has the correct dimensions
    ASSERT_EQ(gradientInput.rows(), inputTensor.rows());
    ASSERT_EQ(gradientInput.cols(), inputTensor.cols());
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
