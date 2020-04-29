# collaborativeDL
边缘设备协同深度学习推理系统，client.py作为任务发起端，server.py作为任务接收端，client完成卷积部分计算，server完成全连接部分计算并返回结果。

rpc框架使用pickle、socket加动态代理模式实现，不需要对推理部分代码做任何改动即可实现远程调用服务端协同计算。
***
TODO
整合注册中心
整合缓存
