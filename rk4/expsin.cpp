#include <iostream>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include <pthread.h>
#include <cstdlib>
#include <cstdio>
using namespace std;
using std::exp;
using std::sin;
using std::cos;
using std::log;
#define NUM_THREADS 100

using dtype = long double; //determines floating pt precision
using dfns = vector<dtype (*)(vector<dtype>)>; //vector of func ptrs

#include "integrators.cpp"

#define PI 3.141592653589793
#define EUL 2.71828182845904523536028747135266249775724709369995957
#define g 9.8
#define L 1.0

//return vector containing every nth element of V
template<class T>
auto nthValues(vector<T> V, int n) -> vector<T>{
    vector<T> result;
    for(unsigned i = 0; i < V.size(); i++)
        if(i%n==0){result.push_back(V[i]);}
    return result;
}

//print results
template<class T>
void printDV(vector<T> v){printf("time:% 5.5f   y: % 5.5f   dy: % 5.5f ",
                                (float)v[0], (float)v[1], (float)v[2]);}


//V is vector <time, y, y'>, at some timestep;
//dy returns derivative of y, d2y returns 2nd derivative of y
auto y(dtype t) -> dtype{return sin(t * exp(cos(t)));}

auto dy(vector<dtype> V) -> dtype{
    return exp(cos(V[0])) * cos(V[0] * exp(cos(V[0]))) * (1-V[0]*sin(V[0]));
}

auto d2y(vector<dtype> V) -> dtype{
    return  exp(cos(V[0])) * cos(V[0] * exp(cos(V[0]))) * (V[0]*sin(V[0])*sin(V[0]) - V[0]*cos(V[0]) - 2*sin(V[0])) - exp(2*cos(V[0])) * (1-V[0]*sin(V[0]))*(1-V[0]*sin(V[0]))*sin(V[0]*exp(cos(V[0])));
}

struct thread_params{
    int tid;
    dfns F;
    vector<vector<dtype>> DV;
    dtype step;
    dtype tmax;
    vector<dtype> exactsol;
};

void *rk4Worker(void *t_arg){
    struct thread_params *p = (struct thread_params*)t_arg;
    vector<vector<dtype>> resultDV = 
        rk4<dtype>((*p).F, (*p).DV, (*p).step, (*p).tmax);

    p->DV = resultDV;

    ofstream resultfile;
    resultfile.open("data/expsinData"+std::to_string((*p).tid)+".csv");
    for(unsigned j = 0; j < resultDV.size(); j++){
        resultfile << std::to_string(resultDV[j][0]) << ",";
        resultfile << std::to_string(resultDV[j][1]) << ",";
        resultfile << std::to_string(resultDV[j][2]) << "\n";
    }
    resultfile.close();

    vector<dtype> exactsol;
    for(dtype t = (*p).DV[0][0]; t < (*p).tmax; t += (*p).step){
        exactsol.push_back(y(t));
    }

    p->exactsol = exactsol;
    
    pthread_exit(NULL);
}

int main(){
    dtype initStep = 0.1;
    vector<dtype> h = {initStep}; 
    int smallestSplit = NUM_THREADS+1;
    for(int i = 2; i < smallestSplit; i++){
        h.push_back(h[0] / i);
    }
    dtype tmax = 99.9;
    
    //DV is vector <time, y, y'>, at each timestep;
    //DV[0] = initial conditions
    vector<vector<dtype>> inits = {{0.0, 0, EUL}};
    vector<vector<dtype>> DV = inits;

    //F is vector of refs to derivative functions
    dfns F = {&dy, &d2y};

    //prepare thread parameters
    pthread_t threads[NUM_THREADS];
    thread_params args[NUM_THREADS];
    for(int i = 0; i < NUM_THREADS; i++){
        args[i].tid = i;
        args[i].F = F;
        args[i].DV = DV;
        args[i].step = h[i];
        args[i].tmax = tmax;
    }
    
    //start rk4 threads
    unsigned counter = 0;
    for(int i = 0; i < NUM_THREADS; i++){
        //DV = inits;
        cout << "starting thread " << 1+counter++ << "/" << smallestSplit-1;
        cout <<  " for stepsize: " << h[i] << endl;
        pthread_create(&threads[i], NULL, rk4Worker, (void *)&args[i]);
    }

    for(int i = 0; i < NUM_THREADS; i++){
        pthread_join(threads[i], NULL);
    }


    for(int j = 0; j < NUM_THREADS-1; j++){
        cout << "step: " << h[j] << endl;
        int c = args[j].exactsol.size() - 1;
        for(int i = c; i > c-3; i--){
            printDV(args[j].DV[i]);
            cout << "exact: " << args[j].exactsol[i]; 
            cout << "  err: " << args[j].exactsol[i] - args[j].DV[i][1] << endl;
        }
        cout << endl;
    }

    return 0;
}
