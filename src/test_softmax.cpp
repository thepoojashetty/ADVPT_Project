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

    std::cout << "Output Probabilities:\n" << outputProbabilities << "\n";

    // Check if the output probabilities sum to approximately 1
    ASSERT_NEAR(outputProbabilities.rowwise().sum().array().maxCoeff(), 1.0, 1e-9);
}

// Test case for the forward pass of the softmax function
TEST_F(SoftmaxTest, ForwardPass2) {
    // Initialize test data within the test case
    Eigen::MatrixXd inputTensor;
    inputTensor.resize(3, 3);
    inputTensor << 1.0, 2.0, 3.0,
                   4.0, 5.0, 6.0,
                   7.0, 8.0, 9.0;

    Eigen::MatrixXd expectedOutput;
    expectedOutput.resize(3, 3);
    expectedOutput << 0.09003057, 0.24472847, 0.66524096,
                      0.09003057, 0.24472847, 0.66524096,
                      0.09003057, 0.24472847, 0.66524096;

    // Create an instance of your Softmax class
    Softmax softmax;

    // Call the forward method with the input tensor
    Eigen::MatrixXd actualOutput = softmax.forward(inputTensor);

    // Check if the actual output matches the expected output within a tolerance
    double tolerance = 1e-6;
    ASSERT_TRUE(actualOutput.isApprox(expectedOutput, tolerance))
        << "Actual output:\n" << actualOutput
        << "\nExpected output:\n" << expectedOutput;
}


TEST_F(SoftmaxTest, BackwardPass) {
    // Create a Softmax object
    Softmax softmax;

    // Input tensor for testing
    Eigen::MatrixXd inputTensor(2, 3);
    inputTensor << 1, 2, 3,
                   4, 5, 6;

    // Perform the forward pass
    Eigen::MatrixXd outputTensor = softmax.forward(inputTensor);

    // Error tensor for testing
    Eigen::MatrixXd errorTensor(2, 3);
    errorTensor << 0.1, -0.2, 0.3,
                   0.4, 0.5, -0.6;

    // Perform the backward pass
    Eigen::MatrixXd gradientInput = softmax.backward(errorTensor);

    // Check the result against expected values
    // The expected values are calculated manually based on the Softmax derivative
    Eigen::MatrixXd expectedGradientInput(2, 3);
    expectedGradientInput << 0.0222222, -0.0444444, 0.0666667,
                         0.0888889, 0.111111, -0.133333;

    ASSERT_TRUE(gradientInput.isApprox(expectedGradientInput, 1e-4));
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
