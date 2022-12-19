import scipy.integrate
from scipy.integrate import RK45, solve_ivp
import numpy as np
from numpy import array as arr
from math import sin,cos
from matplotlib import pyplot as plt
fig,ax = plt.subplots(figsize=(20,10))

def dy(t, y):
    r = arr([y[1], (-9.8)*sin(y[0])])
    return r

def main():
    t0 = 0.0
    y0 = arr([0.0,1.0]) #inits [vel, angle]
    tmax = 10.0
    h = np.arange(0.1, 0.001, -0.001)
    data = []
    for step in h:
        sol = solve_ivp(dy, (t0,tmax), y0, method="RK45",
            first_step=step, max_step=step, rtol=1e99, atol=1e99)
        d = zip(sol.t, sol.y[1])
        for i in d:
            data.append((step,i[0],i[1]))
        plt.plot(sol.t, sol.y[1], label="step="+str(step))
    
    ax.set_xlim([tmax-5,tmax+2])
    plt.legend()
    plt.show()
    #data = np.array(data)
    #np.save("data.npy",data)



main()




