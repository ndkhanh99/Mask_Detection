#----- A simple TCP client program in Python using send() function -----
import socket
import time

while True:
    clientSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM);
    clientSocket.connect(("127.0.0.1", 1234));
    dataFromServer = clientSocket.recv(1024);
    print(dataFromServer.decode());
    data = "Hoan thanh mot chu trinh";
    clientSocket.send(data.encode());