#pragma once

#include <iostream>
#include <vector>
#include <numeric>
#include <fstream>
#include <chrono>
#include "eigenmvn.h"
#include <omp.h>

// Function to find mean.
double find_mean(std::vector<double> &arr, int n)
{
    double sum = std::accumulate(arr.begin(), arr.end(), 0.0f);
    double mean = (double)sum / (double)n;
    return mean;
}
 
// Function to find covariance.
double find_covariance(std::vector<double> &arr1, std::vector<double> &arr2, int n, double mean_arr1, double mean_arr2)
{
    double sum = 0;
    // float mean_arr1 = find_mean(arr1, n);
    // float mean_arr2 = find_mean(arr2, n);
    for (int i = 0; i < n; i++)
        sum = sum + (arr1[i] - mean_arr1) * (arr2[i] - mean_arr2);
    return sum / (double)(n - 1);
}

double pms(Eigen::EigenMultivariateNormal<double> &normX_solver, double isovalue, int num_samples)
{
    auto R = normX_solver.samples(num_samples).transpose();
    // std::cout << "R " << R.rows() << " " << R.cols() << " " << R.coeff(0, 0) << "\n";
    int numCrossings = 0;
    for (int n = 0; n < num_samples; ++n){
        if ((isovalue <= R.coeff(n, 0)) && (isovalue <= R.coeff(n, 1)) && (isovalue <= R.coeff(n, 2)) && (isovalue <= R.coeff(n, 3))){
            numCrossings = numCrossings + 0;
        }else if((isovalue >= R.coeff(n,0)) && (isovalue >= R.coeff(n,1)) && (isovalue >= R.coeff(n,2)) && (isovalue >= R.coeff(n,3))){
            numCrossings = numCrossings + 0;
        }else{
            numCrossings = numCrossings + 1;
            // std::cout << R.coeff(n, 0) << " " << R.coeff(n,1) << " " << R.coeff(n,2) << " " << R.coeff(n,3) << "\n";
        }
    }
    // if (numCrossings == num_samples){
    //     std::cout << "numCrossing: " << numCrossings << " " << (float)numCrossings / num_samples << "\n";
    // }
    
    double crossingProb = (double)numCrossings / num_samples;
    return crossingProb;
}

void cov_matrix(std::vector<std::vector<double>> &data, int size_x, int size_y, int num_members, int num_samples, double isovalue)
{
    
    std::vector<double> probs;
    for(int j = 0; j < size_x - 1; ++j){
        for(int i = 0; i < size_y - 1; ++i){
            int index0 = size_y * j + i;
            int index1 = index0 + size_y;
            int index2 = index0 + 1;
            int index3 = index1 + 1;
            // if (index0 < 100)
            // std::cout << "index0: " << index0 << " ";
            std::vector<double> d0;
            std::vector<double> d1;
            std::vector<double> d2;
            std::vector<double> d3;
           
            for(int m = 0; m < num_members; ++m){
                // std::vector<double> d = data[m];
                d0.push_back(data[m][index0]);
                d1.push_back(data[m][index1]);
                d2.push_back(data[m][index2]);
                d3.push_back(data[m][index3]);
                // if (d[index0] == 0 || d[index1] == 0 || d[index2] == 0 || d[index3] == 0){
                //     std::cout << "m: " << m << "\n";
                //     std::cout << "index: " << index0 << " " << index1 << " " << index2 << " " << index3 << "\n";
                //     std::cout << d[index0] << " " << d[index1] << " " << d[index2] << " " << d[index3] << std::endl;
                // }
                // std::cout << data[m][index0] << " " << data[m][index1] << " " << data[m][index2] << " " << data[m][index3] << "\n";
            }
            double mean_0 = find_mean(d0, num_members);
            double mean_1 = find_mean(d1, num_members);
            double mean_2 = find_mean(d2, num_members);
            double mean_3 = find_mean(d3, num_members);
            
            // calculate covariance 
            std::vector<std::vector<double>> d{d0, d1, d2, d3};
            // generate mean and cov matrix
            Eigen::Vector4d mean;
            mean << mean_0, mean_1, mean_2, mean_3;

            std::vector<float> cov_matrix;
            for(int p = 0; p < 4; ++p){
                for(int q = p; q < 4; ++q){
                    float cov = find_covariance(d[p], d[q], num_members, mean[p], mean[q]);
                    cov_matrix.push_back(cov);
                }       
            }  
            Eigen::Matrix4d covar;
            covar << cov_matrix[0], cov_matrix[1], cov_matrix[2], cov_matrix[3],
                     cov_matrix[1], cov_matrix[4], cov_matrix[5], cov_matrix[6],
                     cov_matrix[2], cov_matrix[5], cov_matrix[7], cov_matrix[8],
                     cov_matrix[3], cov_matrix[6], cov_matrix[8], cov_matrix[9];
              
            // if (index0 == 1){
            //     std::cout << covar << "\n";
            // }
            
            // const int dim = 4;
            Eigen::EigenMultivariateNormal<double> normX_solver(mean, covar);
            // std::ofstream file_solver("samples_solver.txt");
            // file_solver << normX_solver.samples(100).transpose() << std::endl;
            double p = pms(normX_solver, isovalue, num_samples);
            probs.push_back(p);
            // if(mean_0 == 0 || mean_1 == 0 || mean_2 == 0 || mean_3 == 0){
            //     std::cout << mean_0 << " " << mean_1 << " " << mean_2 << " " << mean_3 << "\n";
            //     std::cout << p << "\n";
            // }
            // if (p == 1){
            //     std::cout << "P: " << p << std::endl;
            // }
        }
        
    }

    // Debug: 
    float min_val = *std::min_element(probs.begin(), probs.end());
    float max_val = *std::max_element(probs.begin(), probs.end());
    std::cout << "Min: " << min_val << "\n";
    std::cout << "Max: " << max_val << "\n";
    

}