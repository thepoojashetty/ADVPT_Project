#pragma once

#include "Eigen/Dense"

class Relu
{
private:
    Eigen::MatrixXd inputTensorCache;

public:
    Relu();
    ~Relu();

    Eigen::MatrixXd forward(const Eigen::MatrixXd &);
    Eigen::MatrixXd backward(const Eigen::MatrixXd &);
};

Relu::Relu() {}

Relu::~Relu() {}

Eigen::MatrixXd Relu::forward(const Eigen::MatrixXd &inputTensor)
{
    inputTensorCache = inputTensor;
    return inputTensor.array().cwiseMax(0);
}

Eigen::MatrixXd Relu::backward(const Eigen::MatrixXd &errorTensor)
{
    return (inputTensorCache.array() <= 0).select(0, errorTensor);
}