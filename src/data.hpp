#pragma once

#include <iostream>
#include <memory>
#include <fstream>
#include <vector>
#include "Eigen/Dense"

class DataSetImages
{
private:
    //specify the batch size during object initialization
    size_t batch_size_;
    //
    size_t number_of_images_;
    size_t number_of_rows_;
    size_t number_of_columns_;
    std::vector<Eigen::MatrixXd> batches_;
public:
    DataSetImages(size_t batch_size);
    ~DataSetImages();
    void readImageData(const std::string&);
    void writeImageToFile(const std::string&,const size_t&);
    Eigen::MatrixXd getBatch(const size_t&);
    size_t getNoOfBatches();
};

DataSetImages::DataSetImages(size_t batch_size)
{
    batch_size_=batch_size;
}

DataSetImages::~DataSetImages(){}

Eigen::MatrixXd DataSetImages::getBatch(const size_t& index)
{
    return batches_[index];
}

size_t DataSetImages::getNoOfBatches()
{
    return batches_.size();
}

//Reads file containing MNIST image data and stores it in batches
//Makes use of Eigen library to store the data in a matrix
void DataSetImages::readImageData(const std::string& input_filepath)
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

        //read number of images
        int number_of_images=0;
        input_file.read(bin_data,4);
        std::reverse(bin_data,bin_data+4); //reverse magic number
        std::memcpy(&number_of_images, bin_data, sizeof(int));
        // std::cout<<"number of images: "<<number_of_images<<std::endl;
        number_of_images_ = number_of_images;
    
        //read number of rows
        int number_of_rows=0;
        input_file.read(bin_data,4);
        std::reverse(bin_data,bin_data+4); //reverse magic number
        std::memcpy(&number_of_rows, bin_data, sizeof(int));
        // std::cout<<"number of rows: "<<number_of_rows<<std::endl;
        number_of_rows_ = number_of_rows;
    
        //read number of columns
        int number_of_columns=0;
        input_file.read(bin_data,4);
        std::reverse(bin_data,bin_data+4); //reverse magic number
        std::memcpy(&number_of_columns, bin_data, sizeof(int));
        // std::cout<<"number of columns: "<<number_of_columns<<std::endl;
        number_of_columns_ = number_of_columns;

        //read images
        size_t image_size = number_of_rows_*number_of_columns_;
        size_t images_in_last_batch = number_of_images_%batch_size_;
        unsigned char *image_bin = new unsigned char[image_size];
        double *image = new double[image_size];
        Eigen::MatrixXd image_matrix(batch_size_,image_size);
        size_t batch_end_row=batch_size_;
    
        for(size_t i=0;i<number_of_images_;++i)
        {
            input_file.read(reinterpret_cast<char*>(image_bin),image_size);
            std::transform(image_bin,image_bin+image_size,image,[&](unsigned char c){return static_cast<double>(c)/255.0;});
            image_matrix.row(i%batch_size_) = Eigen::Map<Eigen::VectorXd>(image,image_size);
            if((i+1)%batch_size_==0)
            {
                batches_.push_back(image_matrix.block(0,0,batch_end_row,image_size));
            }
            else if(i==number_of_images_-1)
            {
                batch_end_row = images_in_last_batch;
                batches_.push_back(image_matrix.block(0,0,batch_end_row,image_size));
            }
        }
        input_file.close();
    }
    else
    {
        // Handle error: unable to open the file
        std::cout<<"Unable to open file: "<<input_filepath<<std::endl;
    }
}

void DataSetImages::writeImageToFile(const std::string& output_filepath, const size_t& index){
    size_t batch_no = index / batch_size_;
    size_t image_index = index % batch_size_;
    std::ofstream output_file(output_filepath);

    if (output_file.is_open())
    {
        // Write the rank of the tensor to the file
        output_file<<2<<"\n";

        // Write the shape of the tensor to the file
        output_file<<number_of_rows_<<"\n";
        output_file<<number_of_columns_<<"\n";

        // Write the tensor elements to the file
        size_t image_size = number_of_rows_*number_of_columns_;
        for(size_t i=0;i<image_size;++i)
        {
            output_file<<batches_[batch_no](image_index,i)<<"\n";
        }
        output_file.close();
    }
    else
    {
        // Handle error: unable to open the file
        std::cerr << "Error: Unable to open file for writing: " << output_filepath << std::endl;
    }
}
