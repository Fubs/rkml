#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <pthread.h>
#include <sstream>
#include <string>
#include <vector>
#include <thread>
#include <gmp.h>
#include <gmpxx.h>
#include <boost/math/constants/constants.hpp>
#include <boost/multiprecision/cpp_int.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <boost/multiprecision/mpfr.hpp>
#include <boost/multiprecision/mpc.hpp>

#define ORD_2
#define NUM_THREADS 64

using namespace std;
#include "params.hpp"
#include "integrators.hpp"

//return vector containing every nth element of V
template<class T>
vector<T> nthValues(vector<T> V, int n){
    vector<T> result;
    for(unsigned i = 0; i < V.size(); i++){
        if(i%n==0){result.push_back(V[i]);}
    }
    return result;
}


template<int N>
struct A {
    constexpr A() : arr() {
        #pragma unroll
        for (auto i = 0; i <= N; ++i) {
            arr[i] = initStep/(i+1); 
        }
    }
    dtype arr[N];
};

auto all_h = A<smallestSplit>();

template<class T>
void printDV(array<T,ORD+1> V){
    cout << "time: " << to_string(V[0]) << " y': " << to_string(V[1]) << " y: " << to_string(V[2]) << endl;
}
/*
template<class T>
void printDV(array<T,3> v){printf("time:% 5.5f   vel: % 5.5f   ang: % 5.5f\n",
                                (float)v[0], (float)v[1], (float)v[2]);}
                                */

//derivative functions
//V is vector <time, y', y>, at some timestep;
//d2y returns 2nd derivative of y, dy returns derivative of y
//dtype d2y(array<dtype,3> V){return -1*(g/L)*sin(V[2]);}
dtype d2y(array<dtype,3> V){return -V[2];}
dtype dy(array<dtype,3> V){return V[1];}

struct thread_params{
    int tid{};
    dfns F;
    vector<array<dtype,3>> DV;
    dtype step{};
    dtype t{};
} __attribute__((packed)) __attribute__((aligned(64)));

auto rk4Worker(void *t_arg){
    auto *p = static_cast<struct thread_params*>(t_arg);
    vector<array<dtype,3>> resultDV = 
        rk4<dtype>((*p).F, (*p).DV, static_cast<size_t>((*p).t/((*p).step)), (*p).step);

    ofstream resultfile;
    resultfile.open("data/pendulumData"+std::to_string((*p).tid)+".csv");
    for(auto & j : resultDV){
        if(USING_MPFR){
            resultfile << boost::multiprecision::to_string(mpfr_float(j[0])) << ",";
            resultfile << boost::multiprecision::to_string(mpfr_float(j[1])) << ",";
            resultfile << boost::multiprecision::to_string(mpfr_float(j[2])) << ",";
            resultfile << boost::multiprecision::to_string(mpfr_float(j[3])) << "\n";
        }
        else{
            resultfile << std::to_string(static_cast<long double>(j[0])) << ",";
            resultfile << std::to_string(static_cast<long double>(j[1])) << ",";
            resultfile << std::to_string(static_cast<long double>(j[2])) << ",";
            resultfile << std::to_string(static_cast<long double>(j[3])) << "\n";
        }
    }
    resultfile.close();
    return;
}



int main(){
    
    cout << " - main start - " << endl;

    dtype piconst = boost::math::constants::pi<dtype>();

    //DV is vector <time, vel, ang>, at each timestep;
    //DV[0] = initial conditions, angle in radians
    array<dtype,3> inits = {0, PI, 0};
    vector<array<dtype,3>> DV = {inits};

    cout << " - inits set - " << endl;

    //F is vector of pointers to derivative functions
    dfns F = {&d2y, &dy};

    //prepare thread parameters
    thread threads[NUM_THREADS];
    cout << " - threadarrays allocated - " << endl;
    thread_params args[NUM_THREADS];
    cout << " - argarrays allocated - " << endl;

    for(int i = 0; i < NUM_THREADS; i++){
        args[i].tid = i+1;
        args[i].F = F;
        args[i].DV = DV;
        args[i].step = all_h.arr[i];
        args[i].t = tmax;
    }
    //order threads from smallest to largest step size (largest to smallest run time)
    std::reverse(std::begin(args), std::end(args));
    cout << " - args built - " << endl;

    
    //start rk4 threads 
    unsigned counter = 0;
    #pragma unroll (NUM_THREADS)
    for(int i = 0; i < NUM_THREADS; i++){
        //DV = inits;
        cout << "starting thread " << 1+counter++ << "/" << smallestSplit;
        cout <<  " for stepsize: " << all_h.arr[i] << endl;
        threads[i] = thread(rk4Worker, &args[i]);
    }

    vector<vector<array<dtype,3>>*> results;

    for(int i = 0; i < NUM_THREADS; i++){
        threads[i].join();
        results.push_back(&args[i].DV);
    }

    dtype exact = piconst*static_cast<dtype>(boost::math::sin_pi(tmax/piconst));
    cout << "exact solution: " << exact << endl;
    size_t c = 0;
    //print results
    for(auto & i : results){
        if (i->back()[0] == dtype(20)){
            auto *tmp = i;
            vector<array<dtype,3>> tmp2 = {tmp->back()};
            auto newDV = (rk4<dtype>(F, tmp2, 1, (*i)[0][0] - (*i)[1][0]));
            i->push_back(newDV[0]);
        }
    }



    for(auto & i : results){
        //print first and last
        cout << "difference from exact: " << i->back()[2] - exact << " ";
        printDV(i->back());
    }

    return 0;
}
