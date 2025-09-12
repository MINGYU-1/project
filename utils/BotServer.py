import socket
class BotServer:
    def __init__(self, srv_port, listen_num):
        self.port = srv_port
        self.listen = listen_num
        self.mySock = None
        
    def create_sock(self):
        self.mySock = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        self.mySock.bind(('0.0.0.0', int(self.port)))
        self.mySock.listen(int(self.listen))
        return self.mySock
    
    # Client 대기
    def ready_for_client(self):
        return self.mySock.accept()
    #sock 변환
    def get_sock(self):
        return self.mySock.accept()
    
