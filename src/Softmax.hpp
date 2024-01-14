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

// Should use Eigen library to implement the forward and backward pass

class Softmax {
private:
    Eigen::MatrixXd input_;
    Eigen::MatrixXd output_;
    Eigen::MatrixXd error_;
public:
    Softmax();
    ~Softmax();
    Eigen::MatrixXd forward(Eigen::MatrixXd);
    Eigen::MatrixXd backward(Eigen::MatrixXd, Eigen::MatrixXd);
};

Softmax::Softmax() {}

Softmax::~Softmax() {}

Eigen::MatrixXd Softmax::forward(Eigen::MatrixXd input) {
    input_ = input;
    output_ = input_.array().exp();
    output_ = output_ / output_.sum();
    return output_;
}

Eigen::MatrixXd Softmax::backward(Eigen::MatrixXd error, Eigen::MatrixXd target) {
    error_ = error;
    error_ = error_ - target;
    return error_;
}

