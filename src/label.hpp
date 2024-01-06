#pragma once

#include <iostream>
#include <memory>
#include <fstream>
#include <vector>
#include "Eigen/Dense"

// Class to store MNIST label data
class DatasetLabels
{
private:
    size_t batch_size_;
    size_t number_of_labels_;
    std::vector<Eigen::MatrixXd> batches_;
public:
    DatasetLabels(size_t batch_size);
    ~DatasetLabels();
    void readLabelData(const std::string&);
    // void writeLabelToFile(const std::string&,const size_t&);
};

DatasetLabels::DatasetLabels(size_t batch_size)
{
    batch_size_=batch_size;
}

DatasetLabels::~DatasetLabels(){}

//Reads file containing MNIST label data and stores it in batches
//Makes use of Eigen library to store the data in a matrix
void DatasetLabels::readLabelData(const std::string& input_filepath)
{
    std::ifstream input_file(input_filepath,std::ios::binary);
    if(input_file.is_open())
    {
        //read magic number
        char bin_data[4];
        int magic_number;
        input_file.read(bin_data,4);
        std::reverse(bin_data,bin_data+4); //reverse magic number
        std::memcpy(&magic_number, bin_data, sizeof(int));
        // std::cout<<"magic number: "<<magic_number<<std::endl;

        //read number of labels
        int number_of_labels=0;
        input_file.read(bin_data,4);
        std::reverse(bin_data,bin_data+4); //reverse magic number
        std::memcpy(&number_of_labels, bin_data, sizeof(int));
        // std::cout<<"number of labels: "<<number_of_labels<<std::endl;
        number_of_labels_ = number_of_labels;
    
        //read label data
        int label=0;
        for(size_t i=0;i<number_of_labels_;i++)
        {
            input_file.read(bin_data,1);
            std::memcpy(&label, bin_data, sizeof(int));
            Eigen::MatrixXd label_matrix(1,10);
            label_matrix.setZero();
            label_matrix(0,label)=1;
            batches_.push_back(label_matrix);
        }
    }
    else
    {
        std::cout<<"Unable to open file"<<std::endl;
    }
}

/*
// Untested
//Writes label data to file
void DatasetLabels::writeLabelToFile(const std::string& output_filepath,const size_t& index)
{
    // Write in this format to be compatible with this format
    // 1
    // 10
    // 0 if label is 0
    // 1 if label is 1
    // label 0, 1 should be written for respective digit entry
    std::ofstream output_file(output_filepath,std::ios::binary);
    if(output_file.is_open())
    {
        //write magic number
        int magic_number = 2049;
        char bin_data[4];
        std::memcpy(bin_data, &magic_number, sizeof(int));
        std::reverse(bin_data,bin_data+4); //reverse magic number
        output_file.write(bin_data,4);

        //write number of labels
        int number_of_labels = 1;
        std::memcpy(bin_data, &number_of_labels, sizeof(int));
        std::reverse(bin_data,bin_data+4); //reverse magic number
        output_file.write(bin_data,4);

        //write label
        Eigen::MatrixXd label_matrix = batches_[index];
        int label = 0;
        for(size_t i=0;i<10;i++)
        {
            if(label_matrix(0,i)==1)
            {
                label = i;
                break;
            }
        }
        std::memcpy(bin_data, &label, sizeof(int));
        output_file.write(bin_data,1);
    }
    else
    {
        std::cout<<"Unable to open file"<<std::endl;
    }

    
}
*/