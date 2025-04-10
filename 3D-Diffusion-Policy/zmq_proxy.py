import zmq 
 
context = zmq.Context()
 
# 前端套接字（ROUTER接收客户端请求）
frontend = context.socket(zmq.ROUTER) 
frontend.bind("tcp://192.168.110.200:8899")   # 绑定本地端口 
 
# 后端套接字（DEALER转发到远程服务器）
backend = context.socket(zmq.DEALER) 
backend.connect("tcp://172.23.255.227:8000")  # 替换为实际远程地址 
 
# 启动代理 
zmq.proxy(frontend,  backend)