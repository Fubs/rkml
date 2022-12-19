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
#include <semaphore>
#include <gmp.h>
#include <gmpxx.h>
#include <boost/math/constants/constants.hpp>
#include <boost/multiprecision/cpp_int.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <boost/multiprecision/mpfr.hpp>
#include <boost/multiprecision/mpc.hpp>
#include <sciplot/sciplot.hpp>

#define ORD_3

using namespace sciplot;
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
} __attribute__((packed)) __attribute__((aligned(32)));

auto all_h = A<NUM_THREADS>();

void padTo(std::string &str, const size_t num, const char paddingChar = ' '){
    if(num > str.size()) {
        str.insert(0, num - str.size(), paddingChar);
    }
}
template<class T>
void printDV(array<T,ORD+1> V){
    //std::string str0 = std::to_string(V[0]); std::string str1 = to_string(V[1]); std::string str2 = to_string(V[2]); std::string str3 = to_string(V[3]);
    std::string str0 = to_string(V[0]); std::string str1 = to_string(V[1]); std::string str2 = to_string(V[2]); std::string str3 = to_string(V[3]);
    padTo(str0, 10); padTo(str1, 9); padTo(str2, 9); padTo(str3, 9);
    cout << "time: " << str0 << "   V_c1: " << str1 << " V_c2: " << str2 << " i_L: " << str3 << endl;
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
} __attribute__((packed)) ;

//constexpr size_t batch_size = 2;
//std::counting_semaphore<batch_size> batch_sem(batch_size);

auto rk4Worker(void *t_arg){
    //batch_sem.acquire();

    auto *p = static_cast<struct thread_params*>(t_arg);
    vector<array<dtype,ORD+1>> resultDV = 
        rk4<dtype>((*p).F, (*p).DV, static_cast<size_t>(std::round(tmax/((*p).step))), (*p).step);

    ofstream resultfile;
    resultfile.open("data/chua"+to_string((*p).tid)+".csv");
    //vector<array<dtype,ORD+1>> resultDV2 = nthValues<array<dtype,ORD+1>>(resultDV, ((*p).tid));
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
    //batch_sem.release();
    return;
}


#define G dtype(0.7)
#define C1_inv dtype(9)
#define C2_inv dtype(1)
#define L_inv dtype(7)
#define m0 dtype(-0.5)
#define m1 dtype(-0.8)
#define Bp dtype(1)

inline dtype chua_diode_resp(dtype v) {return m0*v + dtype(0.5)*dtype((m1)-(m0))*abs(v+Bp) + dtype(0.5)*dtype((m0)-(m1))*abs(v-Bp);}


// DV = [t, v_c1, v_c2, i_L]
// d/dt( v_c1 ) = G*C1_inv*(v_c2-v_c1) - C1_inv*chua_diode_resp(v_c1)
inline dtype dv_c1(array<dtype,4> V){return G*C1_inv*(V[2]-V[1]) - C1_inv*chua_diode_resp(V[1]);} 

// d/dt( v_c2 ) = G*C2_inv*(v_c1-v_c2) + C2_inv*i_L 
inline dtype dv_c2(array<dtype,4> V){return G*C2_inv*(V[1]-V[2]) + C2_inv*V[3];}

// d/dt( i_L ) = -L_inv * v_c2
inline dtype dv_c3(array<dtype,4> V){return -L_inv*V[2];}

int main(){
    
    cout << " - main start - " << endl;

    //DV is vector <time, z, y, x>, at each timestep;
    //DV[0] = initial conditions
    array<dtype,4> inits = {dtype(0), dtype(-0.02281), dtype(0.38127),  dtype(0.15264)};
    vector<array<dtype,4>> DV = {inits};

    cout << " - inits set - " << endl;

    //F is vector of pointers to derivative functions
    dfns F = {&dv_c1, &dv_c2, &dv_c3};

    //prepare thread parameters
    thread threads[NUM_THREADS];
    cout << " - threadarrays allocated - " << endl;
    //thread_params args[NUM_THREADS];
    vector<thread_params> args = vector<thread_params>(NUM_THREADS);
    cout << " - args allocated - " << endl;

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
    cout << " - starting threads - " << endl;
    #pragma unroll
    for(int i = NUM_THREADS-1; i >= 0; i--){
        //DV = inits;
        cout << "\rstarting thread " << 1+counter++ << "/" << smallestSplit << " for stepsize: " << all_h.arr[i] << flush;
        threads[i] = thread(rk4Worker, &args[i]);
    }

    cout << endl << " - joining threads - " << endl;

    vector<vector<array<dtype,4>>*> results;

    for(int i = 0; i < NUM_THREADS; i++){
        threads[i].join();
        cout << "\rthreads done: [" << i+1 << "/" << NUM_THREADS << "]" << flush;
    }
    cout << endl << " - threads joined - " << endl;

    for(int i = 0; i < NUM_THREADS; i++){
        results.emplace_back(&args[i].DV);
    }

    //dtype exact = piconst*dtype(boost::math::sin_pi(tmax/piconst));
    //cout << "exact solution: " << exact << endl;
    //print results
    int c = NUM_THREADS-1;
    for(auto & i : results){
        //print first and last
        auto ss = std::to_string((long double)(all_h.arr[c--]));
        padTo(ss, 10);
        cout << "timestep: " << ss << " ";
        printDV(i->back());
        /*
        size_t j = 1;
        while(j++){
            cin.get();
            printDV(i->at(j));
        }
        printDV(i->back());
        cin.get();
        */
    }

    ofstream hfile;
    hfile.open("data/h.csv");
    for(auto & j : all_h.arr){
        hfile << std::to_string((long double)(j)) << "\n";
    }
    hfile.close();



    return 0;
}
