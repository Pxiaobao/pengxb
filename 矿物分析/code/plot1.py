import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import pdb
rock_character = pd.read_excel('../data/模型数据副本.xlsx',sheet_name = '材料')
# 设置图像大小
plt.figure(figsize=(10, 8))
plt.rcParams['font.sans-serif'] = ['SimHei', 'Songti SC', 'STFangsong']
plt.rcParams['axes.unicode_minus'] = False
cols = ['设备S', '设备Z', '静态抗压强度', '弹性模量', '泊松比', '抗拉强度','黏聚力', '内摩擦角', '拉强比', '脆性指数', '回弹均值', '动态强度', '滑动摩擦系数', '声级',
       '波速', '密度均值', '渗透率', '孔隙度', '粒径']

rock_character['内摩擦角'].astype(float)
for col in cols:
    data = rock_character[col].values

    plt.hist(data,  color='green', edgecolor='black')
    plt.title(col)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    #plt.show()
    plt.savefig('../output直方图/' + col + '.png', dpi=300, bbox_inches='tight')
    plt.clf()

import configparser
import socket     #导入socket模块
from fastapi import FastAPI, Request, Response, Body	
import uvicorn as u
import time
import json
import requests
import httpx
def opendoorids():
    print('准备连接')
    s = socket.socket()                     #创建套接字
    global host                             #IP
    global port                             #端口
    global storeId
    s.connect((host,int(port)))                 #主动初始化TCP服务器连接
    s.settimeout(1)
    s.send(b"\xa0\x04\x01\x89\x01\xd1") 
    res = []
    print('开始接收数据')
    try:
        recvData = s.recv(1024)
        EPC = recvData[7:-4]
        result = hex(int.from_bytes(EPC, byteorder='big',signed=False))
        if result in res:
            pass
        else:
            #print(result)
            res.append(result)
    except Exception as e:
        #输出超时信息
        print(e)
    print('接收数据结束')
    #关闭套接字
    s.close()

    res = [_format(i) for i in res]
    res = list(filter(lambda x: x is not None and x.strip() != "" and x!='0' and len(x)>10 and len(x)<23, res))
    res = list(set(res))
    res_data = {
        "dataList":res,
        "storeId":int(storeId),
        "count":len(res)
    }
    print(res)


config = configparser.ConfigParser()
config.read('D:\jwx\gate_machine\conf.ini')
url = config['web']['url']
host = config['door']['ip']
port = config['door']['port']
storeId = config["door"]['storeid']
#每隔1秒钟调用一次opendoorids函数
while True:
    opendoorids()
    time.sleep(1)

