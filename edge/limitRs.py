import os
import time

import psutil
import resource
import sys
try:
    print('---------')
    print(os.getpid())
    print(resource.getrlimit(resource.RLIMIT_AS))
    print('---------')
    # 错误 1024 ** 4 -> 1B
    # 正确 unit B
    resource.setrlimit(resource.RLIMIT_AS, (0.57 * 1024 ** 3, 0.57 * 1024 ** 3))
    print('~~~~~~~~~')
    print(resource.getrlimit(resource.RLIMIT_AS))
    print('~~~~~~~~~')

    print('memory used {}'.format(round(psutil.Process(os.getpid()).memory_full_info().rss, 2)))
    nums = []
    for n in range(10 ** 8):
        # nums.append(n)
        nums.append("a")

    print("ok")
    print(sys.getsizeof(nums))
    print(len(nums))
    print('memory used {}'.format(round(psutil.Process(os.getpid()).memory_full_info().rss, 2)))
    print('==========')

    print(resource.getrlimit(resource.RLIMIT_AS))
    print('==========')
finally:
    print('111111')
    print(sys.getsizeof(nums))
    print(len(nums))
    print(resource.getrlimit(resource.RLIMIT_AS))
    print('memory used {}'.format(round(psutil.Process(os.getpid()).memory_full_info().rss, 2)))
    time.sleep(60)
    print('memory used {}'.format(round(psutil.Process(os.getpid()).memory_full_info().rss, 2)))