#pragma once

class SGD{
    private:
        double learningRate;
    public:
        SGD();
        SGD(double);
        ~SGD();

        Eigen::MatrixXd updateWeights(Eigen::MatrixXd &, Eigen::MatrixXd &);
};

SGD::SGD(){
    this->learningRate = 0.001;
}

SGD::SGD(double learningRate){
    this->learningRate = learningRate;
}

SGD::~SGD(){}

Eigen::MatrixXd SGD::updateWeights(Eigen::MatrixXd &weights, Eigen::MatrixXd &gradient){
    return (weights - learningRate * gradient);
}
