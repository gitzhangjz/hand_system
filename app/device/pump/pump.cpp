/*x86_64-w64-mingw32-g++.exe pump.cpp  -fPIC -shared -o a.so -m64 生成64位的库*/


#include<cstdio>
#include<windows.h>
#include <ctime>
#include <string>
#include <iostream>
#include <cstring>
// #include <Python.h> 

extern "C"
{
class pump{
	HANDLE hCom;

	//*设置超时  GetCommTimeouts/SetCommTimeouts
	//*设置缓冲区大小	SetupComm()
	//* 设置串口配置信息  GetCommState()和SetCommState() 
	int setUart()
	{
		COMMTIMEOUTS timeouts;
		DCB dcb;
		
		//读超时 
		timeouts.ReadIntervalTimeout = 1000;		//读操作时两个字符间的间隔超时
		timeouts.ReadTotalTimeoutMultiplier = 500;	//读操作在读取每个字符时的超时
		timeouts.ReadTotalTimeoutConstant = 5000;	//读操作的固定超时
		//写超时 
		timeouts.WriteTotalTimeoutMultiplier = 0;	//写操作在写每个字符时的超时
		timeouts.WriteTotalTimeoutConstant = 2000;	//写操作的固定超时
		
		SetCommTimeouts(hCom,&timeouts);
		
		//设置输入输出缓冲区大小
		SetupComm(hCom,500,500);
		
		//设置串口参数，如波特率 
		if (GetCommState(hCom, &dcb) == 0)
		{
			return -1;
		}
		
		dcb.BaudRate = CBR_9600;	//波特率 
		dcb.ByteSize = 8;			//数据位数 
		dcb.Parity = NOPARITY;		//校验位 
		dcb.StopBits = ONESTOPBIT;	//停止位 
		
		if(SetCommState(hCom,&dcb) == 0)
		{
			return -1;
		}
		
		return 0;
		
	}

	// 参数必须为整型，表示延时多少秒
	void delay(int seconds) 
	{
		clock_t start = clock();
		clock_t lay = (clock_t)seconds * CLOCKS_PER_SEC;
		while ((clock()-start) < lay);
	}

public:
	/*
		act：{bend : 抓握(气泵吹气0x05), extend : 张开(气泵吸气0x03)}
		s:秒数
	*/
	void blow(const std::string& act, int s)
	{
		char wbuf[10] = {0};
		if(act == "extend")
			wbuf[0] = 0x05;
		else if(act == "bend")
			wbuf[0] = 0x03;
		else {
			wbuf[0] = 0x00;
			std::cerr << "wrong arg" << std::endl;
			return; 
		}

		DWORD wsize = 0;
		WriteFile(hCom, wbuf, 1, &wsize, NULL);
		delay(s);

		//关闭开关
		wbuf[0] = 0x00;
		WriteFile(hCom, wbuf, 1, &wsize, NULL);
		debug();

	}

	//接收传回的数据
	void debug()
	{
		char rbuf[1024] = {0};
		DWORD rsize = 0;
		ReadFile(hCom,rbuf,1024,&rsize,NULL);
	}

	//手套连接电脑串口的名称
	pump(const std::string &s)
	{
		// std::wstring s(uart_name.cbegin(), uart_name.cend());
		hCom = CreateFile(s.c_str() ,GENERIC_READ |GENERIC_WRITE, 0,NULL,OPEN_EXISTING,0,NULL);
		if (hCom !=INVALID_HANDLE_VALUE)
		{
			std::cerr << "Open Succes!" << std::endl;
		}else 
		{ 
			std::cerr << "Open Failed!" << std::endl;
		} 
		
		//配置串口 
		if(setUart() == -1)
		{
			if(INVALID_HANDLE_VALUE != hCom)
			{
				CloseHandle(hCom);	//关闭串口 
				std::cerr << "setUart failed" << std::endl;
			}
		} 
	}
};

pump *p = nullptr;

void init(char com[])
{
	// std::cout << std::string(com) << std::endl;
	p = new pump(com);
}

void act(char s[], int sec)
{
	// std::cout << std::string(s) << "," << sec << std::endl;
	(*p).blow(std::string(s), sec);
}
// }
// int main()
// {
// 	pump p("COM5");
// 	std::string s;
// 	int sec;
// 	while(1)
// 	{
// 		std::cin >> s >> sec;
// 		// act：{bend : 抓握(气泵吹气0x03), extend : 张开(气泵吸气0x05)}
// 		// s:秒数
// 		if(s == "stop")
// 			break;
// 		p.blow(s, sec);
// 	}

// 	return 0;
}