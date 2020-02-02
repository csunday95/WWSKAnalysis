
from typing import Optional
import socket
import select
import json
from sk_monitor import SKMonitor
from threading import Thread, Lock


class SKMonitorServer:
    def __init__(self, host_port: int, monitor: SKMonitor):
        self._host_port = host_port
        self._monitor = monitor
        self._server_socket = None  # type: Optional[socket.socket]
        self._accept_thread = None  # type: Optional[Thread]
        self._handle_clients_thread = None  # type: Optional[Thread]
        self._serving = False
        self._client_lock = Lock()
        self._connected_clients = []

    def start_server(self):
        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.bind(('127.0.0.1', self._host_port))
        self._server_socket.listen(4)
        self._accept_thread = Thread(target=self._accept_callback)
        self._handle_clients_thread = Thread(target=self.handle_clients_callback)
        self._serving = True
        self._accept_thread.start()
        self._handle_clients_thread.start()

    def stop_server(self):
        self._serving = False
        self._server_socket.close()
        if self._accept_thread is not None:
            self._accept_thread.join()
        
    def _accept_callback(self):
        while self._serving:
            try:
                new_conn, addr = self._server_socket.accept()
            except socket.error as e:
                return
            new_conn.settimeout(0)
            with self._client_lock:
                self._connected_clients.append(new_conn)

    def generate_client_command(self, command):
        if 'cmd' in command:
            cmd_typ = command['cmd']
        else:
            return None
        if cmd_typ == 'board':
            return self._monitor.get_last_board().as_json()
        elif cmd_typ == 'reset':
            self._monitor.reset_game_start()
            return json.dumps(True)

    def handle_clients_callback(self):
        while self._serving:
            recv_sockets, _, _ = select.select(self._connected_clients, [], [], 0.1)
            if len(recv_sockets) == 0:
                continue
            for sock in recv_sockets:
                try:
                    client_data = sock.recv(1024)
                except socket.error:
                    continue
                if len(client_data) == 0:
                    continue
                client_data = json.loads(client_data)
                resp = self.generate_client_command(client_data)
                if resp is not None:
                    sock.send(resp.encode())
