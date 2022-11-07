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
double find_mean(double* arr, int &n)
{
    double sum = 0;
    for(int i = 0; i < n; ++i){
        sum += arr[i];
    }
    double mean = sum / n;
    return mean;
}
 
// Function to find covariance.
double find_covariance(double *arr1, double *arr2, int &n, double &mean_arr1, double &mean_arr2)
{
    double sum = 0;
    for (int i = 0; i < n; i++)
        sum = sum + (arr1[i] - mean_arr1) * (arr2[i] - mean_arr2);
    return sum / (n - 1);
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
    
    // std::cout << " data size : " << data[0].size() << std::endl;
    int max_index = size_y * (size_x - 1 - 1) + (size_y - 1 - 1);
    
    #pragma omp parallel for
    for(int i = 0; i <= max_index; ++i){
        int index_0 = i;
        int index_1 = index_0 + size_y;
        int index_2 = index_0 + 1;
        int index_3 = index_1 + 1;
        double d0[num_members] =  { 0 };
        double d1[num_members] =  { 0 };
        double d2[num_members] =  { 0 };
        double d3[num_members] =  { 0 };
        for(int m = 0; m < num_members; ++m){
            d0[m] = data[m][index_0];
            d1[m] = data[m][index_1];
            d2[m] = data[m][index_2];
            d3[m] = data[m][index_3];
        }
        double mean_0 = find_mean(d0, num_members);
        double mean_1 = find_mean(d1, num_members);
        double mean_2 = find_mean(d2, num_members);
        double mean_3 = find_mean(d3, num_members);
        // std::cout << "mean: " << mean_0 << " " << mean_1 << " " << mean_2 << " " << mean_3 << "\n";
        Eigen::Vector4d mean;
        mean << mean_0, mean_1, mean_2, mean_3;
        Eigen::Matrix4d covar;
        float cov = find_covariance(d0, d0, num_members, mean_0, mean_0);
        covar(0, 0) = cov;
        cov = find_covariance(d0, d1, num_members, mean_0, mean_1);
        covar(0, 1) = cov;
        covar(1, 0) = cov;
        cov = find_covariance(d0, d2, num_members, mean_0, mean_2);
        covar(0, 2) = cov;
        covar(2, 0) = cov;
        cov = find_covariance(d0, d3, num_members, mean_0, mean_3);
        covar(0, 3) = cov;
        covar(3, 0) = cov;

        cov = find_covariance(d1, d1, num_members, mean_1, mean_1);
        covar(1, 1) = cov;
        cov = find_covariance(d1, d2, num_members, mean_1, mean_2);
        covar(1, 2) = cov;
        covar(2, 1) = cov;
        cov = find_covariance(d1, d3, num_members, mean_1, mean_3);
        covar(1, 3) = cov;
        covar(3, 1) = cov;

        cov = find_covariance(d2, d2, num_members, mean_2, mean_2);
        covar(2, 2) = cov;
        cov = find_covariance(d2, d3, num_members, mean_2, mean_3);
        covar(2, 3) = cov;
        covar(3, 2) = cov;

        cov = find_covariance(d3, d3, num_members, mean_3, mean_3);
        covar(3, 3) = cov;

        
        // std::cout << i << std::endl;
        // std::cout << covar << "\n";
        

        Eigen::EigenMultivariateNormal<double> normX_solver(mean, covar, true);
        auto R = normX_solver.samples(num_samples).transpose();

    }// end of for loop i
}
