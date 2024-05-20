from app.device.Neuracle_API_PY.neuracle_lib.dataServer import DataServerThread
import numpy as np
import time,threading
import matplotlib.pyplot as plt
import pickle
import random

def read_data(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

class data_geter(object):
    '''
        dev = 'eeg'是干电极脑电帽
        dev = 'emg'是肌电设备
    '''
    def __init__(self, dev, time_buffer=3) -> None:
        self.state = 'unlink'
        
        # 配置设备
        # neuracle = dict(device_name='Neuracle', hostname='127.0.0.1', port=8712,
        #                 srate=1000, chanlocs=['Pz', 'POz', 'PO3',  'PO4', 'PO5', 'PO6', 'Oz', 'O1', 'O2', 'TRG'], n_chan=10)
        dsi = dict(device_name='DSI', hostname='127.0.0.1', port=8844,
                srate=300,
                chanlocs=['P3', 'C3', 'F3', 'Fz', 'F4', 'C4', 'P4', 'Cz', 'CM', 'A1', 'Fp1', 'Fp2', 'T3', 'T5', '01',
                            'O2', 'X3', 'X2', 'F7', 'F8', 'X1', 'A2', 'T6', 'T4', 'TRG'], n_chan=25)
        # neuroscan = dict(device_name='Neuroscan', hostname='127.0.0.1', port=4000,
        #                 srate=1000,
        #                 chanlocs=['Pz', 'POz', 'PO3', 'PO4', 'PO5', 'PO6', 'Oz', 'O1', 'O2', 'TRG'] + ['TRG'], n_chan=65)
        emg8 = dict(device_name='Emg8', hostname='127.0.0.1', port=8712,
                    srate=1000, chanlocs=['S01', 'S02', 'S03', 'SO4', 'SO5', 'SO6', 'S07', 'S08', 'trigger'], n_chan=9)

        # dsi是脑电设备
        device = [dsi, emg8]
        if dev == "eeg":
            devicenum = 0  # 脑电
        elif dev == "emg":
            devicenum = 1  # 肌电

        target_device = device[devicenum]

        # 设备名称
        self.dev = dev
        # 采样频率
        self.fq = target_device['srate']
        # 通道数,最后一道为trigger
        self.n_chan = target_device['n_chan']

        # 初始化 DataServerThread 线程
        self.thread_data_server = DataServerThread(device=target_device['device_name'], n_chan=target_device['n_chan'],
                                            srate=target_device['srate'], t_buffer=time_buffer)
        
        # 建立TCP/IP连接
        notconnect = self.thread_data_server.connect(hostname=target_device['hostname'], port=target_device['port'])
        if notconnect:
            raise TypeError("Plz open the hostport,can't connect recorder")
        else:
            # 启动线程
            self.state = 'linked'
            self.thread_data_server.Daemon = True
            self.thread_data_server.start()
            print(dev+' data server connected')
        
        # 数据容器
        self.data = np.empty((self.n_chan,0))
        self.labels = []
        self.trigers = []

        # 是否记录数据的标记，True：数据开始充入self.data。 False：self.data为空
        self.recording_flag = False
        self.running_flag = True
        self.t = threading.Thread(target=self.running)
        self.t.start()

    def running(self):
        while self.running_flag:
            nUpdate = self.thread_data_server.GetDataLenCount()

            if nUpdate > (1 * self.fq - 1):
                tmp_data = self.thread_data_server.GetBufferData()[:, -nUpdate:]
                # print(tmp_data.shape)
                self.thread_data_server.ResetDataLenCount()
                if self.recording_flag:
                    self.data = np.hstack([self.data, tmp_data])
            # print("size : ", self.data.shape)
            time.sleep(0.2)
    
    '''
        开始记录数据, 记录之前清空数据
    '''
    def start_record(self):
        self.recording_flag = True

    def triger(self, tg):
        self.labels.append(tg)
        self.trigers.append(self.data.shape[1])

    '''
        停止记录
    '''
    def stop_record(self):
        self.recording_flag = False
    
    '''
        取出(数据, label, trigger)
    '''
    def get_data(self):
        return self.data, np.array(self.labels), np.array(self.trigers)
    
    def set_label(self, label):
        self.trigers.append(self.data.shape[1])
        self.labels.append(label)

    def clear(self):
        self.data = np.empty((self.n_chan,0))
        self.labels = []
        self.trigers = []

    def stop(self):
        self.running_flag = False
        self.thread_data_server.stop()
    
    # def cocculate(self):
    #     # plt.plot(self.data[0])
    #     # plt.show()
    #     return self.data.shape[1]

    def get_state(self):
        return self.state
    

class fake_data_geter(object):
    '''
        dev = 'eeg'是干电极脑电帽
        dev = 'emg'是肌电设备
    '''
    def __init__(self, dev, time_buffer=3) -> None:
        self.state = 'linked'
        self.array = [[int(random.random()*100) for _ in range(1000)] for _ in range(5)]
        self.trans_indx = 0
        # 配置设备
        # neuracle = dict(device_name='Neuracle', hostname='127.0.0.1', port=8712,
        #                 srate=1000, chanlocs=['Pz', 'POz', 'PO3',  'PO4', 'PO5', 'PO6', 'Oz', 'O1', 'O2', 'TRG'], n_chan=10)
        dsi = dict(device_name='DSI', hostname='127.0.0.1', port=8844,
                srate=300,
                chanlocs=['P3', 'C3', 'F3', 'Fz', 'F4', 'C4', 'P4', 'Cz', 'CM', 'A1', 'Fp1', 'Fp2', 'T3', 'T5', '01',
                            'O2', 'X3', 'X2', 'F7', 'F8', 'X1', 'A2', 'T6', 'T4', 'TRG'], n_chan=25)
        # neuroscan = dict(device_name='Neuroscan', hostname='127.0.0.1', port=4000,
        #                 srate=1000,
        #                 chanlocs=['Pz', 'POz', 'PO3', 'PO4', 'PO5', 'PO6', 'Oz', 'O1', 'O2', 'TRG'] + ['TRG'], n_chan=65)
        emg8 = dict(device_name='Emg8', hostname='127.0.0.1', port=8712,
                    srate=1000, chanlocs=['S01', 'S02', 'S03', 'SO4', 'SO5', 'SO6', 'S07', 'S08', 'trigger'], n_chan=9)

        # dsi是脑电设备
        device = [dsi, emg8]
        if dev == "eeg":
            devicenum = 0  # 脑电
        elif dev == "emg":
            devicenum = 1  # 肌电

        target_device = device[devicenum]

        # 设备名称
        self.dev = dev
        # 采样频率
        self.fq = target_device['srate']
        # 通道数,最后一道为trigger
        self.n_chan = target_device['n_chan']

        # 数据容器
        import os
        self.data = read_data(os.path.join(os.getcwd(), 'app', 'data', dev+'_fake.pkl'))

    def transfer_data(self):
        
        # 生成3*100个随机数
        random_array = [x[self.trans_indx:self.trans_indx+100] for x in self.array[:]]
        self.trans_indx = (self.trans_indx + 1)%900
        return random_array
    
    def start_record(self):
        pass

    def stop_record(self):
        pass
    
    def get_data(self):
        return self.data,[],[]
    
    def clear(self):
        pass

    def stop(self):
        pass

    def get_state(self):
        return self.state
    def set_label(self, label):
        pass
class fake_Pump(object):
    def __init__(self):
        pass
    def act(self, a, b):
        print('act:',a,b,' ')