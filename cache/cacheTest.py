from __future__ import absolute_import

import torch
import pickle
import numpy as np
from cache.MyCache import LRUCache
import redis

conn = redis.Redis(host='10.4.10.228', port=6379)
print(conn.hgetall('cache'))
cache = conn.hgetall('cache')

for i in cache:
    print(cache[i])
# lru = LRUCache(5)
# lru.cacheWarming()
# print(lru)
#
# key = np.random.rand(512,7,7).reshape((1*512*7*7))
# print(key)
# key = torch.from_numpy(key)
#
# print(key)
# conn.hset("test1",key=pickle.dumps(key),value=1)
# # conn.hset("test",key='key',value='value')
# print(key.shape)
# # key = pickle.dumps(key)
# # lru.put(key,10)
# for i in range(5):
#
#
#     lru.put(key,10)
#
#     print(lru.size)
#
#
#
#
# sim = lru.cos_sim(key, key)
# print(sim)