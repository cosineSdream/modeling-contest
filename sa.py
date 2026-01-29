import numpy as np
import random
import matplotlib.pyplot as plt

# 城市坐标
N_CITY = 20
cities = np.random.rand(N_CITY,2)

def path_length(path):
    dist = 0
    for i in range(len(path)):
        j = (i+1)%len(path)
        dist += np.linalg.norm(cities[path[i]] - cities[path[j]])
    return dist

# ======================
# SA 参数
# ======================
T = 10.0
T_min = 1e-3
alpha = 0.95
L = 100   # 每个温度下扰动次数

# ======================
# 初始化解
# ======================
cur = random.sample(range(N_CITY), N_CITY)
cur_len = path_length(cur)
best = cur[:]
best_len = cur_len

record = []

# ======================
# SA 主循环
# ======================
while T > T_min:
    for _ in range(L):
        # 生成邻域解 (交换两个城市)
        new = cur[:]
        a,b = random.sample(range(N_CITY),2)
        new[a], new[b] = new[b], new[a]
        
        new_len = path_length(new)
        dE = new_len - cur_len
        
        if dE < 0 or random.random() < np.exp(-dE/T):
            cur, cur_len = new, new_len
            
            if cur_len < best_len:
                best, best_len = cur[:], cur_len
    
    record.append(best_len)
    T *= alpha

print("Best length:", best_len)

plt.plot(record)
plt.xlabel("Temperature step")
plt.ylabel("Best length")
plt.title("SA Convergence Curve")
plt.show()
