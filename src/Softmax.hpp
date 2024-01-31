#pragma once

#include "Eigen/Dense"

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
    Eigen::MatrixXd gradientInput = yHat.array() * (1.0 - yHat.array());
    gradientInput = gradientInput.cwiseProduct(errorTensor);

    return gradientInput;
}
