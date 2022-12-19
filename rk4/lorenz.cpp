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

#define ORD_3

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
        #pragma unroll (N)
        for (auto i = 0; i <= N; ++i) {
            arr[i] = initStep/(i+1); 
        }
    }
    dtype arr[N];
} __attribute__((aligned(128)));

auto all_h = A<smallestSplit>();

template<class T>
void printDV(array<T,ORD+1> V){
    cout << "time: " << to_string(V[0]) << " z: " << to_string(V[1]) << " y: " << to_string(V[2]) << " x: " << to_string(V[3]) << endl;
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

struct thread_params{
    int tid{};
    dfns F;
    vector<array<dtype,ORD+1>> DV;
    dtype step{};
    dtype t{};
} __attribute__((aligned(128)));

auto rk4Worker(void *t_arg){
    auto *p = static_cast<struct thread_params*>(t_arg);
    vector<array<dtype,ORD+1>> resultDV = 
        rk4<dtype>((*p).F, (*p).DV, static_cast<size_t>((*p).t/((*p).step)), (*p).step);

    ofstream resultfile;
    resultfile.open("data/lorenz"+to_string((*p).tid)+".csv");
    vector<array<dtype,ORD+1>> resultDV2 = nthValues<array<dtype,ORD+1>>(resultDV, ((*p).tid));
    cout << "thread " << (*p).tid << " finished, writing to file " << "data/lorenz"+to_string((*p).tid)+".csv" << endl;
    for(auto & j : resultDV2){
        /*
        resultfile << std::to_string(j[0]) << ",";
        resultfile << std::to_string(j[1]) << ",";
        resultfile << std::to_string(j[2]) << "\n";
        */
        resultfile << boost::multiprecision::to_string(j[0]) << ",";
        resultfile << boost::multiprecision::to_string(j[1]) << ",";
        resultfile << boost::multiprecision::to_string(j[2]) << ",";
        resultfile << boost::multiprecision::to_string(j[3]) << "\n";
    }
    resultfile.close();
    return;
}


#define rho 28
#define sigma 10
#define beta (8.0/3.0)
dtype dz(array<dtype,4> V){return V[3]*V[2]-beta*V[1];}
dtype dy(array<dtype,4> V){return V[3]*(rho-V[1])-V[2];}
dtype dx(array<dtype,4> V){return sigma*(V[2]-V[3]);}

int main(){
    
    cout << " - main start - " << endl;

    //DV is vector <time, z, y, x>, at each timestep;
    //DV[0] = initial conditions
    array<dtype,4> inits = {dtype(0), dtype(1)/100, dtype(-1.0),  dtype(20)};
    vector<array<dtype,4>> DV = {inits};

    cout << " - inits set - " << endl;

    //F is vector of pointers to derivative functions
    dfns F = {&dz, &dy, &dx};

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

    
    //start rk4 threads in reverse order (longest runtime first
    unsigned counter = 0;
    #pragma unroll
    for(int i = NUM_THREADS-1; i >= 0; i--){
        //DV = inits;
        cout << "starting thread " << 1+counter++ << "/" << smallestSplit;
        cout <<  " for stepsize: " << all_h.arr[i] << endl;
        threads[i] = thread(rk4Worker, &args[i]);
    }

    vector<vector<array<dtype,4>>*> results;

    for(int i = 0; i < NUM_THREADS; i++){
        threads[i].join();
        results.emplace_back(&args[i].DV);
    }

    //dtype exact = piconst*dtype(boost::math::sin_pi(tmax/piconst));
    //cout << "exact solution: " << exact << endl;
    //print results
    for(auto & i : results){
        //print first and last
        cout << "stepsize: " << i->at(1)[0] - i->at(0)[0] << " ";
        printDV(i->back());
    }
    // write list of stepsizes for python to read
    ofstream stepfile;
    stepfile.open("data/lorenz_step_sizes.csv");
    for(auto & i : all_h.arr){
        stepfile << boost::multiprecision::to_string(i) << "\n";
    }




    return 0;
}
