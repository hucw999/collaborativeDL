import numpy as np
import configparser
import pickle
class ListNode:
    def __init__(self, key=None, value=None):
        self.key = key
        self.val = value
        self.prev = None
        self.next = None

conf = configparser.ConfigParser()
conf.read('../conf/inf.conf')
class LRUCache:
    """ 双向链表 + 哈希表  """
    def __init__(self, capacity: int):
        self.head, self.tail = ListNode(0, 0), ListNode(0, 0)
        self.head.next, self.tail.prev = self.tail, self.head
        self.lookup = {} # 记录键与节点映射关系
        self.maxsize = capacity
        self.size = 0

    def delete(self, node): # 删除节点
        self.lookup.pop(node.key)
        node.prev.next, node.next.prev = node.next, node.prev
        self.size = self.size - 1

    def append(self, node): # 插入节点
        self.lookup[node.key] = node
        cur, pre = self.tail, self.tail.prev
        node.next, node.prev = cur, pre
        pre.next, cur.prev = node, node

    def get(self, key) -> int:
        """ 存在移动到末尾并返回键值，否则返回-1 """
        if key in self.lookup:
            node = self.lookup[key]
            self.delete(node)
            self.append(node)
            return node.val
        return -1

    def get_sim(self, key) -> int:
        """ 存在移动到末尾并返回键值，否则返回-1 """
        for tmpKey in self.lookup:
            if self.cos_sim(key,tmpKey) >0.99:
                node = self.lookup[tmpKey]
                self.delete(node)
                self.append(node)
                return node.val
        return -1

    def put(self, key, value: int) -> None:
        """
        - 存在key则移到末尾
        - 否则，内存满，则移除头部
        - 添加新节点
        """
        if key in self.lookup:
            self.delete(self.lookup[key])
        if len(self.lookup) == self.maxsize:
            self.delete(self.head.next)
        self.append(ListNode(key, value))
        self.size = self.size + 1

    def cos_sim(self, vector_a, vector_b):
        """
        计算两个向量之间的余弦相似度
        :param vector_a: 向量 a
        :param vector_b: 向量 b
        :return: sim
        """
        # vector_a = np.mat(vector_a)
        # vector_b = np.mat(vector_b)
        # num = np.dot(vector_a,  np.transpose(vector_b,(2,1,3,0)))
        num = np.dot(vector_a, vector_b.T)
        denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
        cos = num / denom
        sim = 0.5 + 0.5 * cos
        return sim

    def cosine_similarity(self, x, y, norm=False):
        """ 计算两个向量x和y的余弦相似度 """
        # assert len(x) == len(y), "len(x) != len(y)"
        zero_list = [0] * len(x)
        if x == zero_list or y == zero_list:
            return float(1) if x == y else float(0)

        # method 1
        res = np.array([[x[i] * y[i], x[i] * x[i], y[i] * y[i]] for i in range(len(x))])
        cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))

        # method 2
        # cos = bit_product_sum(x, y) / (np.sqrt(bit_product_sum(x, x)) * np.sqrt(bit_product_sum(y, y)))

        # method 3
        # dot_product, square_sum_x, square_sum_y = 0, 0, 0
        # for i in range(len(x)):
        #     dot_product += x[i] * y[i]
        #     square_sum_x += x[i] * x[i]
        #     square_sum_y += y[i] * y[i]
        # cos = dot_product / (np.sqrt(square_sum_x) * np.sqrt(square_sum_y))

        return 0.5 * cos + 0.5 if norm else cos

    def cacheWarming(self):
        import redis
        host = conf.get('redis', 'host')
        port = conf.getint('redis', 'port')
        conn = redis.Redis(host=host, port=port)

        caches = conn.hgetall("cache")
        for key in caches:
            keyTensor = pickle.loads(key)
            self.put(keyTensor, caches[key])

