import ctypes
from ctypes import c_char_p
from ctypes import c_int32
import os
class Pump:
	# 初始化 com为手套连接电脑的串口号。形式为：COMx
	def __init__(self, com):
		# os.add_dll_directory('C:\\Users\\Administrator\\Desktop\\hand_system-master\\app\\device\\pump')
		ll = ctypes.cdll.LoadLibrary
		self.pump = ll('C:\\Users\\Administrator\\Desktop\\hand_system-master\\app\\device\\pump\\a.so')
		self.pump.init.argtypes = [c_char_p]
		self.pump.act.argtypes = [c_char_p, c_int32]
		self.pump.init(com.encode())
	
	# action：{bend : 抓握(气泵吹气0x03), extend : 张开(气泵吸气0x05)}
	# sec : 动作时间
	def act(self, action, sec):
		self.pump.act(action.encode(), sec)

# p = Pump("COM5")
# p.act("bend", 1)
# p.act("extend", 2)