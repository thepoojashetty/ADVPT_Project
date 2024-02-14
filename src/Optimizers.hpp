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
    // std::cout << "weights: " << weights.row(0).segment(0,7) << std::endl;
    // std::cout << "gradient: " << gradient.row(0).segment(0,7) << std::endl;
    // std::cout<<"learningRate: "<<learningRate<<std::endl;
    return (weights - learningRate * gradient);
}
