#include <iostream>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>

using namespace std;

#define NUM_THREADS 24

typedef long double dtype; //determines floating pt precision
typedef vector<dtype (*)(vector<dtype>)> dfns; //vector of func ptrs

#include "integrators.cpp"

#define PI 3.141592653589793
#define EUL 2.71828182845904523536028747135266249775724709369995957
#define g 9.8
#define L 1.0
#define mu 1.0

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
void printDV(vector<T> v){printf("time:% 5.5f   vel: % 5.5f   ang: % 5.5f\n",
                                (float)v[0], (float)v[1], (float)v[2]);}

//derivative functions
//V is vector <time, y', y>, at some timestep;
//d2y returns 2nd derivative of y, dy returns derivative of y
dtype d2y(vector<dtype> V){return -g-mu*V[1];}
dtype dy(vector<dtype> V){return V[1];}

dtype d2x(vector<dtype> V){return -mu*V[1];}
dtype dx(vector<dtype> V){return V[1];}

struct thread_params{
    int tid;
    dfns F;
    vector<vector<dtype>> DV;
    dtype step;
    dtype tmax;
};

void *rk4Worker(void *t_arg){
    struct thread_params *p = (struct thread_params*)t_arg;
    vector<vector<dtype>> resultDV = 
        rk4<dtype>((*p).F, (*p).DV, (*p).step, (*p).tmax);

    ofstream resultfile;
    resultfile.open("data/pendulumData"+std::to_string((*p).tid)+".csv");
    for(int j = 0; j < resultDV.size(); j++){
        resultfile << std::to_string(resultDV[j][0]) << ",";
        resultfile << std::to_string(resultDV[j][1]) << ",";
        resultfile << std::to_string(resultDV[j][2]) << "\n";
    }
    resultfile.close();
    pthread_exit(NULL);
}

int main(){
    dtype initStep = 0.01;
    vector<dtype> h = {initStep}; 
    int smallestSplit = 25;
    for(int i = 2; i < smallestSplit; i++){
        h.push_back(h[0] / i);
    }
    dtype tmax = 50.0;
    
    //DV is vector <time, vel, pos>, at each timestep;
    //DV[0] = initial conditions
    vector<vector<dtype>> inits = {{0.0, 10.0, 0.0}};
    vector<vector<dtype>> DVY = inits;
    vector<vector<dtype>> DVX = inits;

    //F is vector of pointers to derivative functions
    dfns FY = {&d2y, &dy};
    dfns FX = {&d2x, &dx};

    //prepare thread parameters
    pthread_t ythreads[NUM_THREADS];
    pthread_t xthreads[NUM_THREADS];
    thread_params yargs[NUM_THREADS];
    thread_params xargs[NUM_THREADS];
    for(int i = 0; i < NUM_THREADS; i++){
        yargs[i].tid = i;
        yargs[i].F = FY;
        yargs[i].DV = DVY;
        yargs[i].step = h[i];
        yargs[i].tmax = tmax;

        xargs[i].tid = i;
        xargs[i].F = FX;
        xargs[i].DV = DVX;
        xargs[i].step = h[i];
        xargs[i].tmax = tmax;
    }
    
    //start rk4 threads
    unsigned counter = 0;
    cout << "running y(t) threads" << endl;
    for(int i = 0; i < NUM_THREADS; i++){
        //DV = inits;
        cout << "starting thread " << 1+counter++ << "/" << smallestSplit-1;
        cout <<  " for stepsize: " << h[i] << endl;
        pthread_create(&ythreads[i], NULL, rk4Worker, (void *)&yargs[i]);
    }

    for(int i = 0; i < NUM_THREADS; i++){
        pthread_join(ythreads[i], NULL);
    }

    cout << "running x(t) threads" << endl;
    counter = 0;
    for(int i = 0; i < NUM_THREADS; i++){
        //DV = inits;
        cout << "starting thread " << 1+counter++ << "/" << smallestSplit-1;
        cout <<  " for stepsize: " << h[i] << endl;
        pthread_create(&xthreads[i], NULL, rk4Worker, (void *)&xargs[i]);
    }

    for(int i = 0; i < NUM_THREADS; i++){
        pthread_join(ythreads[i], NULL);
    }




    return 0;
}
