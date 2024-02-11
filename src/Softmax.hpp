#pragma once

#include "Eigen/Dense"
#include <iostream>


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

    // Calculate the softmax of the input tensor
    auto shifted = inputTensor.colwise() - inputTensor.rowwise().maxCoeff();
    Eigen::MatrixXd expTensor = shifted.array().exp();
    yHat = expTensor.array().colwise() / expTensor.array().rowwise().sum();

    return yHat;
}

Eigen::MatrixXd Softmax::backward(const Eigen::MatrixXd &errorTensor)
{
    // Calculate the gradient of the loss with respect to the input
    Eigen::MatrixXd gradientInput = yHat.cwiseProduct(errorTensor);
    auto sigma = gradientInput.rowwise().sum();
    return yHat.cwiseProduct(errorTensor.colwise() - sigma);
}
