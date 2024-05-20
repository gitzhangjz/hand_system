import threading, time

'''
data_geter一共有五种状态:
    1. unlink: 未连接
    2. linked: 已连接，但没有记录数据
    3. recording_data: 正在记录数据
    4. data_ready: 数据已经准备好
    5. coculating: 正在计算
'''

class data_geter(object):
    def __init__(self):
        print("init\n")
        self.data = []
        self.state = 'unlink'

    '''
        连接设备, 数据流动起来但不记录，并设置状态
    '''
    def link(self):
        self.state = 'linked'
        i = 0
        while True:
            if self.state == 'recording_data':
                self.data.append(i)
            if len(self.data) >= 10:
                self.state = 'data_ready'
            i = (i+1)%10
            time.sleep(1)
            print(threading.current_thread().ident)
            print(self.data)
            print(self.state)
            if self.state == 'unlink':
                break
    
    '''
        开始记录数据
    '''
    def get_data(self):
        self.state = 'recording_data'
        
    '''
        停止记录数据，计算分类结果
    '''
    def run(self):
        self.state = 'coculating'

        ret = sum(self.data)
        self.clear()
        res = 'bend' if ret%2==0 else 'extend'
        return res
    
    '''
        清空数据, 状态变为linked
    '''
    def clear(self):
        self.data = []
        self.state = 'run_data'
        return

    '''
        断开连接
    '''
    def unlink(self):
        self.state = 'unlink'
        return

