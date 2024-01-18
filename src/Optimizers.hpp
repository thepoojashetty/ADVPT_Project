#pragma once

class SGD{
    private:
        double learningRate;
    public:
        SGD();
        SGD(double);
        ~SGD();

        void updateWeights(Eigen::MatrixXd &, Eigen::MatrixXd &);
};

SGD::SGD(){
    this->learningRate = 0.001;
}

SGD::SGD(double learningRate){
    this->learningRate = learningRate;
}

SGD::~SGD(){}

void SGD::updateWeights(Eigen::MatrixXd &weights, Eigen::MatrixXd &gradient){
    weights = weights - learningRate * gradient;
}
