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
    Eigen::MatrixXd output = inputTensor.cwiseMax(0.0);
    return output;
}

Eigen::MatrixXd Relu::backward(const Eigen::MatrixXd &errorTensor)
{
    Eigen::MatrixXd mask = (inputTensorCache.array() >= 0.0).cast<double>();
    //the previous fc layer returns one extra element due to the bias term
    Eigen::MatrixXd output = errorTensor.block(0,0,inputTensorCache.rows(),inputTensorCache.cols()).array() * mask.array();
    return output;
}