#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <thread>
#include <sstream>
#include <string>
#include <vector>
#include <boost/math/constants/constants.hpp>
#include <boost/multiprecision/cpp_int.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <boost/multiprecision/mpfr.hpp>
#include <boost/multiprecision/mpc.hpp>
using namespace std;

#define ORD_2

//typedef long double dtype; //determines floating pt precision
//typedef vector<dtype (*)(vector<dtype>)> dfns; //vector of func ptrs

#include "params.hpp"
#include "integrators.hpp"


#define L dtype(1)

//return vector containing every nth element of V
template<class T>
vector<T> nthValues(vector<T> V, int n){
    vector<T> result;
    for(unsigned i = 0; i < V.size(); i++){
        if(i%n==0){result.push_back(V[i]);}
    }
    return result;
}

//print results
template<class T>
/*
void printDV(vector<T> v){printf("time:% 5.5f   vel: % 5.5f   ang: % 5.5f\n",
                                (float)v[0], (float)v[1], (float)v[2]);}
                                */
void printDV(array<T,ORD+1> V){
    cout << "time: " << to_string(V[0]) << " y': " << to_string(V[1]) << " y: " << to_string(V[2]) << endl;
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
} ;

auto all_h = A<NUM_THREADS>();


//derivative functions
//V is vector <time, y', y>, at some timestep;
//d2y returns 2nd derivative of y, dy returns derivative of y
auto d2y(array<dtype,3> V){return dtype(-1*( lil_g / L)) * boost::math::sin_pi(V[2]);}
auto dy(array<dtype,3> V){return V[1];}

struct thread_params{
    int tid{};
    dfns F;
    vector<array<dtype,ORD+1>> DV;
    dtype step;
    dtype t;
} ;

void *rk4Worker(void *t_arg){
    auto *p = static_cast<struct thread_params*>(t_arg);

    vector<array<dtype,ORD+1>> resultDV = 
        rk4<dtype>((*p).F, (*p).DV, size_t(initStepCount * ((*p).tid +1)), (*p).step);
    ofstream resultfile;
    resultfile.open("data/pendulumData"+std::to_string((*p).tid)+".csv");
    if(write_to_file){
        //cout << "thread " << (*p).tid << " finished, writing to file " << "data/chua"+to_string((*p).tid)+".csv" << endl;
        for(auto & j : resultDV){
            if(USING_MPFR){
                resultfile << boost::multiprecision::to_string(mpfr_float(j[0])) << ",";
                resultfile << boost::multiprecision::to_string(mpfr_float(j[1])) << ",";
                resultfile << boost::multiprecision::to_string(mpfr_float(j[2])) << ",";
                resultfile << boost::multiprecision::to_string(mpfr_float(j[3])) << "\n";
            }
            else{
                resultfile << std::to_string((long double)(j[0])) << ",";
                resultfile << std::to_string((long double)(j[1])) << ",";
                resultfile << std::to_string((long double)(j[2])) << ",";
                resultfile << std::to_string((long double)(j[3])) << "\n";
            }
        }
    }
    resultfile.close();
    pthread_exit(nullptr);
}

int main(){
    //vector<dtype> all_h = {initStep}; 
    //for(int i = 2; i < smallestSplit; i++){
        //h.push_back(h[0] / i);
    //}
    
    //DV is vector <time, vel, ang>, at each timestep;
    //DV[0] = initial conditions, angle in radians
    
    array<dtype,3> inits = {0, 0, dtype(PI - dtype(1)/10000)};
    vector<array<dtype,3>> DV = {inits};

    //F is vector of pointers to derivative functions
    dfns F = {&d2y, &dy};

    //prepare thread parameters
    thread threads[NUM_THREADS];
    thread_params args[NUM_THREADS];
    for(int i = 0; i < NUM_THREADS; i++){
        args[i].tid = i+1;
        args[i].F = F;
        args[i].DV = DV;
        args[i].step = dtype(all_h.arr[i]);
        args[i].t = dtype(tmax);
    }
    
    //start rk4 threads
    unsigned counter = 0;
    for(int i = 0; i < NUM_THREADS; i++){
        cout << "starting thread " << 1+counter++ << "/" << NUM_THREADS;
        cout <<  " for stepsize: " << all_h.arr[i] << endl;
        threads[i] = thread(rk4Worker, &args[i]);
    }
    cout << "waiting for threads to finish..." << endl;

    for (int i = 0; i < NUM_THREADS; i++){
        threads[i].join();
        cout << "\rthreads done: [" << i+1 << "/" << NUM_THREADS << "]" << flush;
    }
    cout << endl;

    ofstream hfile;
    hfile.open("data/h.csv");
    for(auto & j : all_h.arr){
        hfile << std::to_string((long double)(j)) << "\n";
    }
    hfile.close();

    cout << "rk4 done" << endl;



    return 0;
}
