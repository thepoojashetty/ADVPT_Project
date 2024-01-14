#pragma once

#include "Eigen/Dense"

// ReLU: rectified linear unit
// Introduces non-linearity to the model
// Outputs the original input for positive values and 0 for negative values
// The number of neurons in the layer is the same as the number of inputs

// Forward pass of the ReLU layer
// f(x) = max(0, x)
// x: input vector
// return: output vector

// Backward pass of the ReLU layer
// Since it is an activation function, we only apply the chain rule to compute the next error tensor as follows
// e(n - 1) = e(n) * (element-wise multiplication) with 1 if x > 0, 0 otherwise
// e(n): error tensor of the current layer
// e(n - 1): error tensor of the previous layer
// return: error tensor of the previous layer

// Should use Eigen library to implement the forward and backward pass

class Relu {
private:
    Eigen::MatrixXd input_;
    Eigen::MatrixXd output_;
    Eigen::MatrixXd error_;
public:
    Relu();
    ~Relu();
    Eigen::MatrixXd forward(Eigen::MatrixXd);
    Eigen::MatrixXd backward(Eigen::MatrixXd);
};

Relu::Relu() {}

Relu::~Relu() {}

Eigen::MatrixXd Relu::forward(Eigen::MatrixXd input) {
    input_ = input;
    output_ = input_.cwiseMax(0);
    return output_;
}

Eigen::MatrixXd Relu::backward(Eigen::MatrixXd error) {
    error_ = error;
    error_ = error_.cwiseProduct(input_.cwiseSign());
    return error_;
}