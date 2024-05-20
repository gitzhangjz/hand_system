from b import *
from b import *
from b import *
from multiprocessing import Process
import time, threading, os,pickle
import matplotlib.pyplot as plt
from random import randint
def read_data(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
if __name__ == '__main__':
    # a = randint(1,2)
    # print(a)
    # eeg =  read_data(os.path.join(os.getcwd(), 'app','data','10', 'eeg_1.pkl'))
    # emg =  read_data(os.path.join(os.getcwd(), 'app','data','10', 'emg_1.pkl'))
    # eeg_trigger =  read_data(os.path.join(os.getcwd(), 'app','data','10', 'eeg_triger_1.pkl'))
    # emg_trigger =  read_data(os.path.join(os.getcwd(), 'app','data','10', 'emg_triger_1.pkl'))
    # label =  read_data(os.path.join(os.getcwd(), 'app','data','10', 'label1.pkl'))
    
    # print(eeg.shape,emg.shape, eeg_trigger, emg_trigger, label)
    # plt.plot(emg[0])
    # plt.show()

    p = os.path.join(os.getcwd(), 'app','data','10', 'eeg_4.pkl')
    print(os.path.exists(p))