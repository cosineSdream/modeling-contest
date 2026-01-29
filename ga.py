import numpy as np
import random
import matplotlib.pyplot as plt

# ======================
# 1. 城市坐标
# ======================
N_CITY = 20
cities = np.random.rand(N_CITY, 2)

def path_length(path):
    dist = 0
    for i in range(len(path)):
        j = (i+1)%len(path)
        dist += np.linalg.norm(cities[path[i]] - cities[path[j]])
    return dist

# ======================
# 2. 初始化种群
# ======================
POP_SIZE = 100
GEN = 300
PC = 0.9
PM = 0.2
ELITE = 2

pop = [random.sample(range(N_CITY), N_CITY) for _ in range(POP_SIZE)]

# ======================
# 3. 选择算子 (锦标赛)
# ======================
def tournament_selection(pop, k=3):
    cand = random.sample(pop, k)
    cand.sort(key=lambda x: path_length(x))
    return cand[0][:]

# ======================
# 4. OX交叉
# ======================
def order_crossover(p1, p2):
    size = len(p1)
    a, b = sorted(random.sample(range(size), 2))
    child = [-1]*size
    child[a:b] = p1[a:b]
    ptr = b
    for x in p2:
        if x not in child:
            if ptr >= size:
                ptr = 0
            child[ptr] = x
            ptr += 1
    return child

# ======================
# 5. 变异算子
# ======================
def swap_mutation(p):
    a, b = random.sample(range(len(p)), 2)
    p[a], p[b] = p[b], p[a]

# ======================
# 6. GA主循环
# ======================
best_record = []

for g in range(GEN):
    new_pop = []

    # 精英保留
    pop.sort(key=lambda x: path_length(x))
    new_pop.extend(pop[:ELITE])

    # 生成其余子代
    while len(new_pop) < POP_SIZE:
        p1 = tournament_selection(pop)
        p2 = tournament_selection(pop)

        if random.random() < PC:
            c1 = order_crossover(p1, p2)
            c2 = order_crossover(p2, p1)
        else:
            c1, c2 = p1[:], p2[:]

        if random.random() < PM:
            swap_mutation(c1)
        if random.random() < PM:
            swap_mutation(c2)   

        new_pop.extend([c1, c2])

    pop = new_pop[:POP_SIZE]

    best_len = path_length(pop[0])
    best_record.append(best_len)

    if g % 50 == 0:
        print(f"Gen {g}, best length = {best_len:.3f}")

# ======================
# 7. 结果展示
# ======================
best_path = pop[0]
print("Final best length:", path_length(best_path))

plt.figure()
plt.plot(best_record)
plt.xlabel("Generation")
plt.ylabel("Best Path Length")
plt.title("GA Convergence Curve")
plt.show()

plt.figure()
x = cities[:,0]; y = cities[:,1]
plt.scatter(x,y)
for i in range(len(best_path)):
    j = (i+1)%len(best_path)
    plt.plot([x[best_path[i]], x[best_path[j]]],
             [y[best_path[i]], y[best_path[j]]])
plt.title("Best Path Found")
plt.show()
