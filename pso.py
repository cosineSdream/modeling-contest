import numpy as np
import matplotlib.pyplot as plt

# ======================
# 目标函数
# ======================
def f(x):
    return x[0]**2 + x[1]**2

# ======================
# 参数
# ======================
N = 30          # 粒子数
ITER = 100
w = 0.7
c1 = 1.7
c2 = 1.7

# ======================
# 初始化
# ======================
pos = np.random.uniform(-10,10,(N,2))
vel = np.random.uniform(-1,1,(N,2))

pbest = pos.copy()
pbest_val = np.array([f(x) for x in pos])

gbest = pbest[np.argmin(pbest_val)]
gbest_val = min(pbest_val)

# ======================
# PSO 主循环
# ======================
curve = []

for t in range(ITER):
    for i in range(N):
        r1, r2 = np.random.rand(), np.random.rand()
        vel[i] = (w*vel[i]
                  + c1*r1*(pbest[i]-pos[i])
                  + c2*r2*(gbest-pos[i]))
        pos[i] = pos[i] + vel[i]

        val = f(pos[i])
        if val < pbest_val[i]:
            pbest[i] = pos[i]
            pbest_val[i] = val

    if min(pbest_val) < gbest_val:
        gbest = pbest[np.argmin(pbest_val)]
        gbest_val = min(pbest_val)

    curve.append(gbest_val)

print("Best solution:", gbest)
print("Best value:", gbest_val)

plt.plot(curve)
plt.xlabel("Iteration")
plt.ylabel("Best f(x)")
plt.title("PSO Convergence Curve")
plt.show()
