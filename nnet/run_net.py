#!/bin/python
from copy import deepcopy
import numpy as np
import tensorflow as tf
import sys, os, time, keras
import keras.models, keras.layers, keras.optimizers, keras.callbacks
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

datapath = 'data/npzips/net_data.npy'

net_data = np.load(datapath)

x_data = np.array(net_data[:,0:2])
y_data = np.array(net_data[:,2:])
t_max = max(x_data[:,0])
print('x_data.shape', x_data.shape)
print('y_data.shape', y_data.shape)
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.25, random_state=42)

model = Sequential()
model_path = 'model.h5'

def make_model() -> Sequential:
    m = Sequential()
    m.add(Dense(256, input_shape=(2,), activation='elu'))
    m.add(Dense(256, activation='elu'))
    m.add(Dense(256, activation='elu'))
    m.add(Dense(256, activation='elu'))
    m.add(Dropout(0.15))
    m.add(Dense(256, activation='elu'))
    m.add(Dense(len(y_data[0]), activation='linear'))
    m.compile(loss='mse', optimizer='adam', metrics=[])
    m.summary()
    return m


def do_train(load_saved=False, save_model=True) -> None:
    m = make_model()
    if load_saved: m.load_weights(model_path)
    batch_size = 128
    epochs = 4
    m.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))
    m.save(model_path)


def do_test():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    m = make_model()
    m.load_weights(model_path)
    test_ts_bins = {}
    for xt,yt in zip(x_test, y_test):
        ts = xt[1]
        if ts not in test_ts_bins: test_ts_bins[ts] = []
        test_ts_bins[ts].append(np.array([xt[0], yt]))
    for i in range(len(test_ts_bins.keys())):
        t_axis = np.linspace(0, 20, 1000)
        ts = list(test_ts_bins.keys())[i]
        inputs = np.column_stack((t_axis, np.full(t_axis.shape, ts)))
        preds = m.predict(inputs)
        y_axis = preds[:,0]
        y2_axis = preds[:,1]
        plt.plot(t_axis, y_axis, label='ts=%s' % ts)
        #plt.plot(t_axis, y2_axis, label='y2')
        if i%10==9:
            plt.legend()
            plt.show()
            input('press enter to continue')
            plt.clf()

def do_sampling():
    import matplotlib.pyplot as plt
    #fig, (ax1,ax2,ax3) = plt.subplots(1,3)
    fig, ax = plt.subplots()
    #make 3 subplots
    m = make_model()
    m.load_weights(model_path)
    test_ts_bins = {}
    for xt,yt in zip(x_test, y_test):
        ts = xt[1]
        if ts not in test_ts_bins: test_ts_bins[ts] = []
        test_ts_bins[ts].append(np.array([xt[0], yt]))
    #samples = np.linspace(0, 1, 4)
    samples = [0.0, 0.25, 0.5,0.75]
    for i in samples:
        t_axis = np.linspace(0, t_max, 1000)
        inputs = np.column_stack((t_axis, np.full(t_axis.shape, i)))
        preds = m.predict(inputs)
        y_axis = preds[:,0] #capacitor 1 voltage
        y2_axis = preds[:,1] #capacitor 2 voltage
        y3_axis = preds[:,2] #current across inductor
        #ax1.plot(t_axis, y_axis, label='ts=%s' % round(i,2))
        #ax2.plot(t_axis, y2_axis, label='ts=%s' % round(i,2))
        #ax3.plot(t_axis, y3_axis, label='ts=%s' % round(i,2))
        plt.plot(t_axis, y_axis, label='timestep=%s' % round(i,2))
    #ax1.legend()
    #ax2.legend()
    #ax3.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Capacitor 1 Voltage (V)')
    plt.title('Chua\'s Circuit: Capacitor 1 Voltage vs Time')
    plt.legend()
    plt.show()


do_train()

#do_all_plots()


#do_test()
do_sampling()














