#include <array>
#include <fstream>
#include <pthread.h>
#include <sstream>
#include <string>
#include <vector>

#include <sciplot/sciplot.hpp>


using namespace std;
using namespace sciplot;
constexpr size_t NUM_THREADS = 3;

using dtype = long double; //determines floating pt precision
using dfns = vector<dtype (*)(vector<dtype>)>; //vector of func ptrs
using sciplot::PI;
#include "integrators.hpp"

constexpr dtype EUL = 2.71828182845904523536028747135266249775724709369995957;
constexpr dtype g = 9.8;
constexpr dtype L = 1.0;

//return vector containing every nth element of V
template<class T>
vector<T> nthValues(vector<T> V, int n){
    vector<T> result;
    //#pragma unroll V.size()
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
dtype d2y(vector<dtype> V){return -1*(g/L)*sin(V[2]);}
dtype dy(vector<dtype> V){return V[1];}

struct thread_params{
    size_t tid;
    dfns F;
    vector<vector<dtype>> DV;
    dtype step;
    dtype tmax;
} __attribute__((aligned(128))) __attribute__((packed));

void* workerThread(void *t_arg){
    auto *p = static_cast<struct thread_params*>(t_arg);
    vector<vector<dtype>> resultDV = 
        verlet<dtype>((*p).F, (*p).DV, (*p).step, (*p).tmax);

    /*
    ofstream resultfile;
    resultfile.open("data/pendulumData"+std::to_string((*p).tid)+".csv");
    for(auto & j : resultDV){
        resultfile << std::to_string(j[0]) << ",";
        resultfile << std::to_string(j[1]) << ",";
        resultfile << std::to_string(j[2]) << "\n";
    }
    resultfile.close();
    */
    pthread_exit(nullptr);
}

int main(){
    cout << "Verlet Integrator" << endl << endl;
    constexpr dtype initStep = 0.01;
    vector<dtype> h; 
    constexpr int smallestSplit = NUM_THREADS;
    //#pragma unroll
    for(int i = 1; i <= NUM_THREADS; i++){
        h.push_back(initStep/ i);
    }
    constexpr dtype tmax = 10.0;
    
    //DV is vector <time, vel, ang>, at each timestep;
    //DV[0] = initial conditions, angle in radians
    vector<vector<dtype>> inits = {{0.0, 0.0, PI-0.0001}};
    vector<vector<dtype>> DV = inits;

    //F is vector of pointers to derivative functions
    dfns F = {&d2y, &dy};

    //prepare thread parameters
    std::array<pthread_t, NUM_THREADS> threads{};
    std::array<struct thread_params, NUM_THREADS> args{};
    size_t c = 0;
    for(auto & i : args){
        i.tid = c;
        i.F = F;
        i.DV = DV;
        i.step = h[c++];
        i.tmax = tmax;
    }
    
    unsigned counter = 0;
    for(int i = 0; i < NUM_THREADS; i++){
        //DV = inits;
        cout << "starting thread " << 1+counter++ << "/" << smallestSplit-1;
        cout <<  " for stepsize: " << h[i] << endl;
        pthread_create(&threads[i], nullptr, workerThread, (void *)&args[i]);
    }

    for(uint64_t thread : threads){
        pthread_join(thread, nullptr);
    }

    cout << "calculation done, making plots" << endl;
    Plot2D angplot;
    Plot2D velplot;
    //angplot.size(800, 800);
    angplot.yrange(-PI, PI);
    angplot.xrange(uint64_t(0), tmax);
    angplot.fontName("Palatino");
    angplot.fontSize(12);
    angplot.xlabel("time");
    angplot.ylabel("angle");
    angplot.legend()
        .atTop()
        .fontSize(10)
        .displayHorizontal()
        .displayExpandWidthBy(2);

    //velplot.size(800, 800);
    velplot.fontName("Palatino");
    angplot.yrange(-1.5, 1.5);
    angplot.xrange(uint64_t(0), tmax);
    velplot.fontSize(12);
    velplot.xlabel("time");
    velplot.ylabel("velocity");
    velplot.legend()
        .atTop()
        .fontSize(10)
        .displayHorizontal()
        .displayExpandWidthBy(2);

    
    cout << "reshaping data" << endl;
    vector<vector<dtype>> times;
    vector<vector<dtype>> angs;
    vector<vector<dtype>> vels;
//#pragma unroll
    for(int i = 0; i < NUM_THREADS; i++){
        vector<dtype> newtime;
        vector<dtype> newang;
        vector<dtype> newvel;
        for(auto & j : args[i].DV){
            newtime.push_back(j[0]);
            newang.push_back(j[2]);
            newvel.push_back(j[1]);
        }
        times.push_back(newtime);
        angs.push_back(newang);
        vels.push_back(newvel);
    }

    cout << "drawing plots" << endl;
    angplot.drawCurve(times[0], angs[0]);
    angplot.drawCurve(times[1], angs[1]);
    angplot.drawCurve(times[2], angs[2]);

    velplot.drawCurve(times[0], vels[0]);
    velplot.drawCurve(times[1], vels[1]);
    velplot.drawCurve(times[2], vels[2]);

    Figure fig = {{angplot, velplot}};
    fig.title("Pendulum");
    Canvas canvas = {{fig}};
    canvas.size(749, 749);
    canvas.save("plot.png");




    return 0;
}
