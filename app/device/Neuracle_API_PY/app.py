import numpy as np
import matplotlib.pyplot as plt

from neuracle_lib.dataServer import DataServerThread
import time

def app():
    # 配置设备
    neuracle = dict(device_name='Neuracle', hostname='127.0.0.1', port=8712,
                    srate=1000, chanlocs=['Pz', 'POz', 'PO3',  'PO4', 'PO5', 'PO6', 'Oz', 'O1', 'O2', 'TRG'], n_chan=10)
    dsi = dict(device_name='DSI', hostname='127.0.0.1', port=8844,
               srate=300,
               chanlocs=['P3', 'C3', 'F3', 'Fz', 'F4', 'C4', 'P4', 'Cz', 'CM', 'A1', 'Fp1', 'Fp2', 'T3', 'T5', '01',
                         'O2', 'X3', 'X2', 'F7', 'F8', 'X1', 'A2', 'T6', 'T4', 'TRG'], n_chan=25)
    neuroscan = dict(device_name='Neuroscan', hostname='127.0.0.1', port=4000,
                     srate=1000,
                     chanlocs=['Pz', 'POz', 'PO3', 'PO4', 'PO5', 'PO6', 'Oz', 'O1', 'O2', 'TRG'] + ['TRG'], n_chan=65)
    emg8 = dict(device_name='Emg8', hostname='127.0.0.1', port=8712,
                srate=1000, chanlocs=['S01', 'S02', 'S03', 'SO4', 'SO5', 'SO6', 'S07', 'S08'], n_chan=8)

    # dsi是脑电设备
    device = [neuracle, dsi, neuroscan, emg8]
    devicenum = 1  # 默认选择emg8设备使用
    
    target_device = device[devicenum]
    # 初始化 DataServerThread 线程
    time_buffer = 3  # second
    thread_data_server = DataServerThread(device=target_device['device_name'], n_chan=target_device['n_chan'],
                                          srate=target_device['srate'], t_buffer=time_buffer)
    # 建立TCP/IP连接
    notconnect = thread_data_server.connect(hostname=target_device['hostname'], port=target_device['port'])
    if notconnect:
        raise TypeError("Plz open the hostport,can't connect recorder")
    else:
        # 启动线程
        thread_data_server.Daemon = True
        thread_data_server.start()
        print('Data server connected')
    # 在线数据获取演示：每隔一秒获取数据（数据长度 = time_buffer）
    N, flagstop = 0, False
    flagstoptimes = 0
    data = None
    time_tuple = time.localtime(time.time())
    print("time : {}".format(time_tuple[5]))
    try:
        while not flagstop:
            nUpdata = thread_data_server.GetDataLenCount()
            time_tuple = time.localtime(time.time())
            if nUpdata > (1 * target_device['srate'] - 1):
                N += 1
                tmp_data = thread_data_server.GetBufferData()
                tmp_data = tmp_data[:,-300:]
                print("nUpdata : {}".format(nUpdata))
                print(tmp_data.shape)
                print("time : {}".format(time_tuple[5]))
                # 拼接data
                data = tmp_data if data is None else np.hstack((data, tmp_data))
                thread_data_server.ResetDataLenCount()
                time.sleep(1)
                # print("单元数据大小：{0}".format(tmp_data.shape))
            
            if N > 59:
                time_str = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
                np.savetxt(time_str+"想象伸展.csv", data, delimiter=',')

                flagstop = True
                flagstoptimes = 0
                print("当前时间为{}年{}月{}日{}点{}分{}秒"
                      .format(time_tuple[0], time_tuple[1], time_tuple[2],
                              time_tuple[3], time_tuple[4], time_tuple[5]))
                print('结束')
            flagstoptimes += 1
    except:
        pass
    # 结束线程
    thread_data_server.stop()


if __name__ == '__main__':
    app()
