#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <fstream>

void load_data(std::string &file_dir, int &m, std::vector<std::vector<double>> &data)
{
    // std::vector<std::vector<float>> data_temp;
    for(int i = 0; i < m; i++){
        std::string filename = file_dir + std::to_string(i) + ".txt";
        std::ifstream fin(filename.c_str());
    
        std::vector<double> d;
        float element;
        while (fin >> element)
        {   
            d.push_back(element);
        }
        data.push_back(d);
    }
    // Normalize data 
    // double min_val = 100000;
    // double max_val = -100000;
    // for(int i = 0; i < data.size(); ++i){
    //     std::vector<double> d = data[i];
    //     double min_temp = *std::min_element(d.begin(), d.end());
    //     double max_temp = *std::max_element(d.begin(), d.end());
    //     if (min_val > min_temp){
    //         min_val = min_temp;
    //     }
    //     if(max_val < max_temp){
    //         max_val = max_temp;
    //     }
    // }
    // std::cout << "min: " << min_val << std::endl;
    // std::cout << "max: " << max_val << std::endl;
    // for(int i = 0; i < data_temp.size(); ++i){
    //     std::vector<float> d_temp = data_temp[i];
    //     std::vector<float> d;
    //     for(int j = 0; j < d_temp.size(); ++j){
    //         float temp = (d_temp[j] - min_val) / (max_val - min_val);
    //         d.push_back(temp);
    //     }
    //     data.push_back(d);
    // }
}