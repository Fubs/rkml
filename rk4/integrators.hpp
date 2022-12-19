#pragma once
#include <iostream>
#include <vector>
#include <array>
#include <algorithm>
#include <boost/multiprecision/gmp.hpp>
using std::vector, std::array, std::cout, std::endl;
                              
#include "params.hpp"

// template argument dtype: data type (float, double, etc...)
// fn arguments: 
//     dfns is vector<dtype (*)(vector<dtype>)>, which is a vector of 
//     function pointers, where each fn takes a vector<dtype> as argument.
//
//     dfns F is the vector of derivative functions for each variable.
//
//     DV stores the values of the variables at each timestep, DV[0] is bc's.
//
//     h is the step size.
//
//     tmax is final time. Initial time is assumed to be 0.
template<class T>
vector<array<T,ORD+1>> rk4(dfns F, vector<array<T,ORD+1>> &DV, const size_t numSteps, const T h) {
    array<T,ORD+1> next{}; std::fill(next.begin(), next.end(), 0);
    array<T,ORD+1> arg{}; std::fill(arg.begin(), arg.end(), 0);
    array<array<T,ORD+1>,4> K = {}; std::fill(K.begin(), K.end(), next);
    DV.reserve(numSteps);
    size_t i = 1;
    #pragma omp parallel for private(arg, K, next)
    for(; i < numSteps; i++){
        std::fill(next.begin(), next.end(), static_cast<dtype>(0));
        std::fill(arg.begin(), arg.end(), static_cast<dtype>(0));
        std::fill(K.begin(), K.end(), array<T,ORD+1>{static_cast<dtype>(0)});
        next[0] = i*h; //next time value = i*stepsize
        #pragma unroll (4)
        for(unsigned k = 0; k < 4; k++){
            arg = DV[i-1];
            // make rk4 arguments to derivative functions
            if(k == 3){
                arg[0] += h;
                #pragma unroll (ORD+1)
                for(unsigned n = 1; n <= ORD+1; n++){
                    arg[n] += K[2][n-1];
                }
            }
            else if(k > 0){
                arg[0] += (h/2);
                #pragma unroll (ORD+1)
                for(unsigned n = 1; n <= ORD+1; n++){
                    arg[n] += K[k-1][n-1]/2;
                }
            }
            //call derivative fn to get rk4 coefficient
            #pragma unroll (ORD)
            for(unsigned j = 0; j < ORD; j++){
                K[k][j] = h*F[j](arg);
            }
        }
        #pragma unroll (ORD)
        for(unsigned j = 1; j < ORD+1; j++){
            next[j]=DV[i-1][j]+(K[0][j-1]+2*K[1][j-1]+2*K[2][j-1]+K[3][j-1])/6;
        }
        DV.emplace_back(next);
    }

    if (DV.back()[0] != tmax) {
        std::fill(next.begin(), next.end(), static_cast<dtype>(0));
        std::fill(arg.begin(), arg.end(), static_cast<dtype>(0));
        std::fill(K.begin(), K.end(), array<T,ORD+1>{static_cast<dtype>(0)});
        next[0] = i*h; //next time value = i*stepsize
        #pragma unroll (4)
        for(unsigned k = 0; k < 4; k++){
            arg = DV[i-1];
            // make rk4 arguments to derivative functions
            if(k == 3){
                arg[0] += h;
                #pragma unroll (ORD+1)
                for(unsigned n = 1; n <= ORD+1; n++){
                    arg[n] += K[2][n-1];
                }
            }
            else if(k > 0){
                arg[0] += (h/2);
                #pragma unroll (ORD+1)
                for(unsigned n = 1; n <= ORD+1; n++){
                    arg[n] += K[k-1][n-1]/2;
                }
            }
            //call derivative fn to get rk4 coefficient
            #pragma unroll (ORD)
            for(unsigned j = 0; j < ORD; j++){
                K[k][j] = h*F[j](arg);
            }
        }
        #pragma unroll (ORD)
        for(unsigned j = 1; j < ORD+1; j++){
            next[j]=DV[i-1][j]+(K[0][j-1]+2*K[1][j-1]+2*K[2][j-1]+K[3][j-1])/6;
        }
        DV.emplace_back(next);

    }

    return DV;
}


//only does 2nd order systems
/*
template<class T>
auto verlet(dfns F, vector<vector<T>> DV, T h, T tmax) -> vector<vector<T>>{
    int numSteps = (int)((tmax - DV[0][0])/h);
    vector<T> next(DV[0].size());
    std::fill(next.begin(), next.end(), 0);
    next[0] = DV[0][0]+h; //time 
    next[2] = DV[0][2] + DV[0][1]*h + (0.5)*F[0](DV[0])*h*h; //position
    next[1] = DV[0][1] + F[0](DV[0])*h; //velocity
    DV.emplace_back(next);
    for(size_t i = 1; i <= numSteps; i++){
        next[0] = i*h;
        for(auto & j : F){
            next[2] = DV[i][2] + DV[i][1]*h + (0.5)*j(DV[i])*h*h;
            next[1] = DV[i][1] + j(DV[i])*h;
            DV.emplace_back(next);
        }
        DV.push_back(next);
    }

    return DV;
}
*/

