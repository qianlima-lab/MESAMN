import time
import multiprocessing
from multiprocessing import Process
import os

def fun(a = 1, b = 2):
  sum = 0
  for i in range(10000000):
    sum += i
  print(sum)
  # print(a, b)

if __name__ == '__main__':
  start = time.time()
  ## 3.5s
  for i in range(10):
    p = Process(target=fun, args=(i, 2))
    p.start()
  p.join()
  
  ## 11.57s
  # for i in range(10):
  #   fun(i, 2)
  print('cost:', time.time() - start)
  


