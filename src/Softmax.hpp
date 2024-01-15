#pragma once

#include "Eigen/Dense"

// Softmax activation function: transforms the output of the last layer into a probability distribution
// The sum of all the elements in the output vector is 1

// Forward pass of the softmax layer
// Encode input neurons x as a probability distribution y using the softmax function
// y = softmax(x) = exp(x) / sum(exp(x))
// x: input vector
// return: output vector

// Backward pass of the softmax layer
// The softmax activation function does not have any parameters to learn
// Therefore, we compute the error tensor of the previous layer as follows
// y' is predicted output, y is actual output
// e(n - 1) = y' - y
// This formula may not be accurate, but it is good enough for the purpose of this project

// Should use Eigen library to implement the forward and backward pass

class Softmax
{
private:
    Eigen::MatrixXd inputTensorCache;
    Eigen::MatrixXd yHat;

public:
    Softmax();
    ~Softmax();

    Eigen::MatrixXd forward(const Eigen::MatrixXd &);
    Eigen::MatrixXd backward(const Eigen::MatrixXd &);
};

Softmax::Softmax() {}

Softmax::~Softmax() {}

Eigen::MatrixXd Softmax::forward(const Eigen::MatrixXd &inputTensor)
{
    inputTensorCache = inputTensor;

    // Subtract the maximum value for numerical stability
    Eigen::MatrixXd shiftedInput = inputTensor.rowwise() - inputTensor.colwise().maxCoeff();

    // Calculate exponentials and the sum of exponentials
    Eigen::MatrixXd exponentials = shiftedInput.array().exp();
    Eigen::MatrixXd exponentSums = exponentials.rowwise().sum().replicate(1, inputTensor.cols());

    // Calculate softmax probabilities
    yHat = exponentials.cwiseQuotient(exponentSums);

    return yHat;
}

Eigen::MatrixXd Softmax::backward(const Eigen::MatrixXd &errorTensor)
{
    // Calculate the gradient of the loss with respect to the input
    Eigen::MatrixXd gradientInput = yHat.array() - errorTensor.array();
    return yHat.cwiseProduct(gradientInput);
}
