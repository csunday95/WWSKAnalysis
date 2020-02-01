import socket
import select
from threading import Thread, Lock


class SKMonitorServer:
    def __init__(self, host_port: int):
        self._host_port = host_port
        self._server_socket = None  # type: socket.socket
        self._accept_thread = None  # type: Thread
        self._serving = False
        self._client_lock = Lock()
        self._connected_clients = []

    def start_server(self):
        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.bind(('localhost', self._host_port))
        self._accept_thread = Thread(target=self._accept_callback)
        self.serving = True
        self._accept_thread.start()

    def stop_server(self):
        self._serving = False
        self._server_socket.close()
        if self._accept_thread is not None:
            self._accept_thread.join()
        
    def accept_callback(self):
        while self._serving:
            try:
                new_conn, addr = self._server_socket.accept()
            except socket.error:
                return
            new_conn.settimeout(0)
            with self.client_lock:
                self._connected_clients.append(new_conn)

    def handle_client_command(self, command):
        if 'cmd' in command:
            cmd_typ = command['cmd']
        else:
            return None
        if cmd_typ == 'board':
            

    def handle_clients_callback(self):
        while self._serving:
            recv_sockets, _, _ = select.select(self._connected_clients, [], [])
            if len(recv_sockets) == 0:
                continue
            for sock in recv_sockets:
                client_data = sock.recv(1024)
                client_data = json.loads(client_data)
                self.handle_client_command(client_data)
        
