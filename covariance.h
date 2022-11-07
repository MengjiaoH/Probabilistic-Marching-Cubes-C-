#pragma once
// #define EIGEN_DONT_PARALLELIZE 
#include <iostream>
#include <vector>
#include <numeric>
#include <fstream>
#include <chrono>
#include "eigenmvn.h"
// #include <omp.h>
#include <tbb/parallel_for.h>

// Function to find mean.
double find_mean(std::vector<double> &arr, int &n)
{
    double sum = std::accumulate(arr.begin(), arr.end(), 0.0f);
    double mean = (double)sum / (double)n;
    return mean;
}
 
// Function to find covariance.
double find_covariance(std::vector<double> &arr1, std::vector<double> &arr2, int &n, double &mean_arr1, double &mean_arr2)
{
    double sum = 0;
    // float mean_arr1 = find_mean(arr1, n);
    // float mean_arr2 = find_mean(arr2, n);
    for (int i = 0; i < n; i++)
        sum = sum + (arr1[i] - mean_arr1) * (arr2[i] - mean_arr2);
    return sum / (double)(n - 1);
}

double pms(Eigen::EigenMultivariateNormal<double> &normX_solver, double &isovalue, int &num_samples, double &find_crossing_time)
{
    // Eigen::setNbThreads(10);
    // Eigen::Matrix<double, Eigen::Dynamic, -1> R;
    // omp_set_dynamic(0); 
    // omp_set_num_threads(128);

    // #pragma omp parallel
    // auto start = std::chrono::high_resolution_clock::now();
    auto R = normX_solver.samples(num_samples).transpose();
    // auto end = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    // sample_time = sample_time + duration.count();
    // std::cout << "R " << R.rows() << " " << R.cols() << " " << R.coeff(0, 0) << "\n";

    // auto start = std::chrono::high_resolution_clock::now();
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
    // auto end = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    // find_crossing_time = find_crossing_time + duration.count();
    // std::cout << "find crossing: " << duration.count() << "\n";
    // if (numCrossings == num_samples){
    //     std::cout << "numCrossing: " << numCrossings << " " << (float)numCrossings / num_samples << "\n";
    // }
    
    double crossingProb = (double)numCrossings / num_samples;
    return crossingProb;
}

void cov_matrix(std::vector<std::vector<double>> &data, int &size_x, int &size_y, int &num_members, int &num_samples, double &isovalue)
{
    auto find_data_time = 0;
    auto cal_mean_time = 0;
    auto cal_cov_time = 0;
    auto get_solver_time = 0;
    auto sample_prob_time = 0;
    double sample_time = 0;
    double find_crossing_time = 0;
    // Eigen::initParallel();
    // omp_set_dynamic(0); 
    // omp_set_num_threads(20);
    #pragma omp parallel for collapse(2)
        // std::vector<double> probs;
    // tbb::parallel_for( tbb::blocked_range<int>(0, size_x-1), [&](tbb::blocked_range<int> r)
    // {
        for(int j = 0; j < size_x -1; ++j){
            // tbb::parallel_for( tbb::blocked_range<int>(0, size_y-1), [&](tbb::blocked_range<int> t){
                for(int i = 0; i < size_y - 1; ++i){
                // auto start = std::chrono::high_resolution_clock::now();
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
                // auto end = std::chrono::high_resolution_clock::now();
                // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                // find_data_time = find_data_time + duration.count();
                // std::cout << "Find Data Cell Time: " << duration.count()  << "\n";
    
                // start = std::chrono::high_resolution_clock::now();
                double mean_0 = find_mean(d0, num_members);
                double mean_1 = find_mean(d1, num_members);
                double mean_2 = find_mean(d2, num_members);
                double mean_3 = find_mean(d3, num_members);
                // end = std::chrono::high_resolution_clock::now();
                // duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                // cal_mean_time = cal_mean_time + duration.count();
                // std::cout << "Calculate Mean Time: " << duration.count() << "\n";
    
                // start = std::chrono::high_resolution_clock::now();
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
                
                // end = std::chrono::high_resolution_clock::now();
                // duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                // cal_cov_time = cal_cov_time + duration.count();
                // std::cout << "Calculate Cov Time: " << duration.count() << "\n";
    
                
                // start = std::chrono::high_resolution_clock::now();
                Eigen::EigenMultivariateNormal<double> normX_solver(mean, covar);
                // auto R = normX_solver.samples(num_samples).transpose();
                // int numCrossings = 0;
                // for (int n = 0; n < num_samples; ++n){
                //     if ((isovalue <= R.coeff(n, 0)) && (isovalue <= R.coeff(n, 1)) && (isovalue <= R.coeff(n, 2)) && (isovalue <= R.coeff(n, 3))){
                //         numCrossings = numCrossings + 0;
                //     }else if((isovalue >= R.coeff(n,0)) && (isovalue >= R.coeff(n,1)) && (isovalue >= R.coeff(n,2)) && (isovalue >= R.coeff(n,3))){
                //         numCrossings = numCrossings + 0;
                //     }else{
                //         numCrossings = numCrossings + 1;
                //         // std::cout << R.coeff(n, 0) << " " << R.coeff(n,1) << " " << R.coeff(n,2) << " " << R.coeff(n,3) << "\n";
                //     }
                // }
                // double p = (double) numCrossings / num_samples;

                // end = std::chrono::high_resolution_clock::now();
                // duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                // get_solver_time = get_solver_time + duration.count();
                // std::cout << "Get Solver Time: " << duration.count() << "\n";
   
                // std::ofstream file_solver("samples_solver.txt");
                // file_solver << normX_solver.samples(100).transpose() << std::endl;

                // auto start = std::chrono::high_resolution_clock::now();
                // double p = pms(normX_solver, isovalue, num_samples, find_crossing_time);
                // auto end = std::chrono::high_resolution_clock::now();
                // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                // sample_prob_time = sample_prob_time + duration.count();
                // std::cout << "Sample Prob Time: " << duration.count() << "\n";
                // probs.push_back(p);
            }
            // });
               
            
        }
    
    // });
        

    // Debug: 
    // float min_val = *std::min_element(probs.begin(), probs.end());
    // float max_val = *std::max_element(probs.begin(), probs.end());
    // std::cout << "Min: " << min_val << "\n";
    // std::cout << "Max: " << max_val << "\n";
    // std::cout << "Find Data Cell Time: " << find_data_time << "\n";
    // std::cout << "Calculate Mean Time: " << cal_mean_time << "\n";
    // std::cout << "Calculate Cov Time: " << cal_cov_time << "\n";
    // std::cout << "Get Solver Time: " << get_solver_time << "\n";
    std::cout << "Sample Prob Time: " << sample_prob_time << "\n";
    std::cout << "sample time: " << sample_time << "\n";
    std::cout << "find crossing time: " << find_crossing_time << "\n";


}