def dfs(k, target, tmp, idx, results):
    if len(tmp) == k:
        if target == 0:
            results.append(tmp)
        return
    
    for i in range(idx, 6):
        if i <= target:
            dfs(k, target - i, tmp + [i], i+1, results)

results = []
n = 8
for k in range(1, n + 1):
    dfs(k, n, [], 1, results)
print(results)