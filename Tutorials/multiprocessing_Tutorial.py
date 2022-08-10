# import os
#
# print('Process (%s) start...' % os.getpid())
# # Only works on Unix/Linux/Mac:
# pid = os.fork()
# if pid == 0:
#     print('I am child process (%s) and my parent is %s.' % (os.getpid(), os.getppid()))
# else:
#     print('I (%s) just created a child process (%s).' % (os.getpid(), pid))

# ##
# import multiprocessing as mp
# from time import sleep
#
#
# print('Before defining simple_func')
#
# def simple_func():
#     print('Starting simple func')
#     sleep(1)
#     print('Finishing simple func')
#
#
# if __name__ == '__main__':
#     p = mp.Process(target=simple_func)
#     p.start()
#     print('Waiting for simple func to end')
#     p.join()

# ##
# import multiprocessing as mp
# from time import sleep
#
# print('Before defining simple_func')
#
# def simple_func():
#     print('Starting simple func')
#     sleep(1)
#     print('Finishing simple func')
#
#
# p = mp.Process(target=simple_func)
# p.start()
# print('Waiting for simple func to end')
# p.join()

##
# import concurrent.futures
# from datetime import datetime
# import time
#
# # start = time.time()
# start = time.perf_counter()
#
#
# def do_something(seconds):
#     print(f'started in {start} second(s)')
#     time.sleep(seconds)
#
# if __name__ == "__main__":
#     with concurrent.futures.ProcessPoolExecutor() as executor:
#         f1 = executor.submit(do_something, 1)
#     print(f'started in {start} second(s)')

# ##
# import multiprocessing
# import threading
# from multiprocessing import Process, Queue
# import os, time, random
#
# # 写数据进程执行的代码:
# def write(q):
#     print('Process to write: %s' % os.getpid())
#     print("process:", os.getpid(), "thread:", threading.get_ident())
#     for value in ['A', 'B', 'C']:
#         print('Put %s to queue...' % value)
#         q.put(value)
#         time.sleep(random.random())
#
# # 读数据进程执行的代码:
# def read(q):
#     print('Process to read: %s' % os.getpid())
#     print("process:", os.getpid(), "thread:", threading.get_ident())
#     while True:
#         value = q.get(True)
#         print('Get %s from queue.' % value)
#
# if __name__=='__main__':
#     # 父进程创建Queue，并传给各个子进程：
#     q = Queue()
#     pw = Process(target=write, args=(q,))
#     pr = Process(target=read, args=(q,))
#     # 启动子进程pw，写入:
#     pw.start()
#     # 启动子进程pr，读取:
#     pr.start()
#     # 等待pw结束:
#     pw.join()
#     # pr进程里是死循环，无法等待其结束，只能强行终止:
#     pr.terminate()

##
import multiprocessing
import threading
from multiprocessing import Process
import os, time, random

def task1():
    print("process1:", os.getpid(), "thread1:", threading.get_ident())

def task2():
    print("process2:", os.getpid(), "thread2:", threading.get_ident())

if __name__=='__main__':
    p1 = Process(target=task1)
    p2 = Process(target=task2)
    print("process0:", os.getpid(), "thread0:", threading.get_ident())
    p1.start()
    p2.start()
    p1.join()
    p2.join()