import psutil

mem = psutil.virtual_memory()

# 系统总计内存
zj = float(mem.total) / 1024 / 1024 / 1024
# 系统已经使用内存
ysy = float(mem.used) / 1024 / 1024 / 1024

# 系统空闲内存
kx = float(mem.free) / 1024 / 1024 / 1024

ava = float(mem.available ) / 1024 / 1024 / 1024

print('系统总计内存:%d.3GB' % zj)
print('系统已经使用内存:%d.3GB' % ysy)
print('系统空闲内存:%d.3GB' % kx)
print('系统空可用内存:%d.3GB' % ava)
