class ListNode:
    def __init__(self, key=None, value=None):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None


class LRUCache:
    """ 双向链表 + 哈希表  """
    def __init__(self, capacity: int):
        self.head, self.tail = ListNode(0, 0), ListNode(0, 0)
        self.head.next, self.tail.prev = self.tail, self.head
        self.lookup = {} # 记录键与节点映射关系
        self.maxsize = capacity

    def delete(self, node): # 删除节点
        self.lookup.pop(node.key)
        node.prev.next, node.next.prev = node.next, node.prev

    def append(self, node): # 插入节点
        self.lookup[node.key] = node
        cur, pre = self.tail, self.tail.prev
        node.next, node.prev = cur, pre
        pre.next, cur.prev = node, node

    def get(self, key: int) -> int:
        """ 存在移动到末尾并返回键值，否则返回-1 """
        if key in self.lookup:
            node = self.lookup[key]
            self.delete(node)
            self.append(node)
            return node.val
        return -1

    def put(self, key:int, value: int) -> None:
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


