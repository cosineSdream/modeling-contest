def knapsack(w, v, W):
    N = len(w)
    dp = [[0]*(W+1) for _ in range(N+1)]
    
    for i in range(1, N+1):
        for j in range(W+1):
            dp[i][j] = dp[i-1][j]
            if j >= w[i-1]:
                dp[i][j] = max(dp[i][j],
                               dp[i-1][j-w[i-1]] + v[i-1])
    return dp[N][W]

w = [2,3,4,5]
v = [3,4,5,6]
print(knapsack(w,v,8))  # 输出最大价值
