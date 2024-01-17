#include <gtest/gtest.h>
#include "Eigen/Dense"
#include "Softmax.hpp"  // Include the header file where Softmax class is defined

class SoftmaxTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize the Softmax instance for each test
        softmaxLayer = Softmax();
    }

    // Declare the Softmax instance
    Softmax softmaxLayer;
};

TEST_F(SoftmaxTest, ForwardPass) {
    // Example input values
    Eigen::MatrixXd inputTensor(3, 3);
    inputTensor << 1.0, 2.0, 3.0,
                   4.0, 5.0, 6.0,
                   7.0, 8.0, 9.0;

    // Forward pass
    Eigen::MatrixXd outputProbabilities = softmaxLayer.forward(inputTensor);

    // Check if the output probabilities sum to approximately 1
    ASSERT_NEAR(outputProbabilities.rowwise().sum().array().maxCoeff(), 1.0, 1e-9);
}

TEST_F(SoftmaxTest, BackwardPass) {
    // Example input values
    Eigen::MatrixXd inputTensor(3, 3);
    inputTensor << 1.0, 2.0, 3.0,
                   4.0, 5.0, 6.0,
                   7.0, 8.0, 9.0;

    // Forward pass (required for the backward pass)
    softmaxLayer.forward(inputTensor);

    // Example error tensor for the backward pass (this should be replaced with actual gradients in a real application)
    Eigen::MatrixXd errorTensor(3, 3);
    errorTensor << 0.1, 0.2, 0.3,
                   -0.4, -0.5, -0.6,
                   0.7, 0.8, 0.9;

    // Backward pass
    Eigen::MatrixXd gradientInput = softmaxLayer.backward(errorTensor);

    // Check if the gradient with respect to input has the correct dimensions
    ASSERT_EQ(gradientInput.rows(), inputTensor.rows());
    ASSERT_EQ(gradientInput.cols(), inputTensor.cols());
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
