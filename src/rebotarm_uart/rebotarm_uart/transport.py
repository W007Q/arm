"""Linux POSIX UART 传输层。

本层只负责 ``/dev/ttyUSB*``、``/dev/serial/by-id/*`` 或测试用 ``/dev/pts``
文件描述符的打开、配置和字节收发，不解释电机指令或 CAN。上层
``link.py`` 决定发送什么消息；``read``/``write`` 只搬运字节。
"""
import errno
import fcntl
import os
import select
import termios
import time


class PosixUart:
    """使用 Linux 原生 termios 的非阻塞 8N1 串口。"""

    def __init__(self, port, baudrate=115200, write_timeout=.02):
        """打开并独占串口；波特率必须是当前系统 termios 支持的值。"""
        speed = getattr(termios, f'B{baudrate}', None)
        if speed is None:
            raise ValueError(f'baudrate {baudrate} is not supported by this platform')
        self.fd = os.open(port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
        self.write_timeout = write_timeout
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            attrs = termios.tcgetattr(self.fd)
            # raw 模式、8 数据位、1 停止位、无软件/硬件流控。
            attrs[0] = attrs[1] = attrs[3] = 0
            attrs[2] = termios.CLOCAL | termios.CREAD | termios.CS8
            attrs[4] = attrs[5] = speed
            # 非阻塞读取；帧边界由 protocol.Decoder 而不是 termios 决定。
            attrs[6][termios.VMIN] = 0
            attrs[6][termios.VTIME] = 0
            termios.tcsetattr(self.fd, termios.TCSANOW, attrs)
            termios.tcflush(self.fd, termios.TCIOFLUSH)
        except BaseException:
            os.close(self.fd)
            raise

    def read(self):
        """读取当前可用的任意字节片段；无数据返回 ``b''``。"""
        if not select.select([self.fd], [], [], 0)[0]:
            return b''
        try:
            data = os.read(self.fd, 4096)
        except BlockingIOError:
            return b''
        if not data:
            raise OSError(errno.EIO, 'UART disconnected')
        return data

    def write(self, packet):
        """完整写出一个已编码帧，处理短写并在超时后抛出异常。"""
        deadline = time.monotonic()+self.write_timeout
        view = memoryview(packet)
        while view:
            remaining = deadline-time.monotonic()
            if remaining <= 0 or not select.select([], [self.fd], [], remaining)[1]:
                raise TimeoutError('UART write timeout')
            try:
                count = os.write(self.fd, view)
            except BlockingIOError:
                continue
            if count <= 0:
                raise OSError('UART write made no progress')
            view = view[count:]

    def close(self):
        """关闭文件描述符；重复关闭是安全的。"""
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1
