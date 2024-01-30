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
    Eigen::MatrixXd output = Eigen::MatrixXd(inputTensor.rows(), inputTensor.cols());
    for (int i=0; i<inputTensor.rows(); i++)
        for (int j=0; j<inputTensor.cols(); j++) {
            if (inputTensor(i,j) >= 0.0)
                output(i, j) = inputTensor(i, j);
            else
                output(i, j) = 0;
        }
    return output;
}

Eigen::MatrixXd Relu::backward(const Eigen::MatrixXd &errorTensor)
{
    Eigen::MatrixXd output = Eigen::MatrixXd(inputTensorCache.rows(), inputTensorCache.cols());
    for (int i=0; i<inputTensorCache.rows(); i++)
        for (int j=0; j<inputTensorCache.cols(); j++) {
            if (inputTensorCache(i,j) >= 0.0)
                output(i, j) = errorTensor(i, j);
            else
                output(i, j) = 0;
        }
    return output;
}