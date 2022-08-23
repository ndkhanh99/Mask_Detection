import socket

# Create a stream based socket(i.e, a TCP socket)

# operating on IPv4 addressing scheme

serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM);


# Bind and listen

serverSocket.bind(("127.0.0.1", 1234))

serverSocket.listen(5);

# Accept connections

while True:

    (clientConnected, clientAddress) = serverSocket.accept();

    print("Accepted a connection request from %s:%s"%(clientAddress[0], clientAddress[1]));

    dataFromClient = clientConnected.recv(1024)
    
    if dataFromClient == b"HelloServer" or b"Mask Detected":
        print(dataFromClient.decode());
        print(clientConnected)
        print(clientAddress)
        clientConnected.sendto("ok".encode('utf-8'), ("127.0.0.1", 8080))
    else:
        clientConnected.close();

    # Send some data back to the client

    
    
    