#include <iostream>
#include <string>
#include <vector>
#include <chrono>

#include "load_data.h"
#include "covariance.h"

int main(int argc, char** argv)
{
    std::string file_dir = "/Users/hanmj/Desktop/Uncertainty/LCP_C++/datasets/txt_files/wind_pressure_200/Lead_33_";
    int num_members = 15;
    int num_samples = 100;
    double isovalue = 0.2;

    // Load data 
    std::vector<std::vector<double>> data;
    load_data(file_dir, num_members, data);
    // Debug
    // for(int i = 0; i < data.size(); ++i){
    //     std::cout << "member " << i << " size: " << data[i].size() << "\n";
    //     for(int j = 0; j < data[i].size(); ++j){
    //         if (j < 10)
    //             std::cout << data[i][j] << " ";
    //     }
    //     std::cout << "\n";
    // }
    int size_x = 121;
    int size_y = 240;
    auto start = std::chrono::high_resolution_clock::now();
    cov_matrix(data, size_x, size_y, num_members, num_samples, isovalue);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Runtime: " << duration.count() << " ms " << std::endl;
    return 0;
}