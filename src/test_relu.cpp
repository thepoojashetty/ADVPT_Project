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
    // Create a ReLU object
    Relu relu;

    // Input tensor for testing (replace with your own values)
    Eigen::MatrixXd inputTensor(2, 2);
    inputTensor << 1, -2,
                   3, 0;

    // Perform the forward pass
    Eigen::MatrixXd outputTensor = relu.forward(inputTensor);

    // Error tensor for testing
    Eigen::MatrixXd errorTensor(2, 2);
    errorTensor << 2, -1,
                   1, 3;

    // Perform the backward pass
    Eigen::MatrixXd gradientInput = relu.backward(errorTensor);

    // Check the result against expected values
    // The expected values are calculated manually based on the ReLU derivative
    Eigen::MatrixXd expectedGradientInput(2, 2);
    expectedGradientInput << 2, 0,
                             1, 0;

    std::cout << "Actual Result:\n" << gradientInput << "\n";
    std::cout << "Expected Result:\n" << expectedGradientInput << "\n";

    ASSERT_TRUE(gradientInput.isApprox(expectedGradientInput, 1e-6));
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
