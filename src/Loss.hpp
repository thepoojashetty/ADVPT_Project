#pragma once

class CrossEntropyLoss {
    private:
        Eigen::MatrixXd predTensorCache;
    public:
        CrossEntropyLoss();
        ~CrossEntropyLoss();

        double forward(const Eigen::MatrixXd &, const Eigen::MatrixXd &);
        Eigen::MatrixXd backward(const Eigen::MatrixXd &);
};

CrossEntropyLoss::CrossEntropyLoss() {}

CrossEntropyLoss::~CrossEntropyLoss() {}

double CrossEntropyLoss::forward(const Eigen::MatrixXd &inputTensor, const Eigen::MatrixXd &labelTensor)
{
    predTensorCache = inputTensor;
    return -((labelTensor.array() * inputTensor.array().log()).sum());
}

Eigen::MatrixXd CrossEntropyLoss::backward(const Eigen::MatrixXd &labelTensor)
{
    // target output/predicted output
    return -(labelTensor.array() / predTensorCache.array());
}