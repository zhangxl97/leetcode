# # 1
# import math
# s, p = map(float, input().split())

# delta = s * s - 4 * p
# if delta >= 0:
#     n1 = (s + math.sqrt(delta)) / 2
#     if n1 > 0:
#         m1 = p / n1
#     else:
#         m1 = -1
#     n2 = (s - math.sqrt(delta)) / 2
#     if n2 > 0:
#         m2 = p / n2
#     else:
#         m2 = -1
#     if (m1 >= 1 and n1 >= 1 and m1 + n1 == s) or (m2 >= 1 and n2 >= 1 and m2 + n2 == s):
#         print("Yes")
#     else:
#         print("No")
# else:
#     print("No") 

# 2
# n = int(input())
# s = input()
# cnt = 0
# start = 0
# index = s.find("fox", start)
# while index != -1:
#     start = max(0, index - 2)
#     s = s[:index] + s[index + 3:]
#     cnt += 1
#     index = s.find("fox", start)
# print(n - cnt * 3)

#3
import sys
class Solver():
    def __init__(self, n, aa, ab, ba, bb):
        self.cnt = 0
        self.kvs = {"AA":aa, "AB":ab, "BA": ba, "BB": bb}
        self.n = n
        self.s = "AB"
        self.strs = []

    def solve(self):
        if n == 2 or n == 3 or self.kvs["AA"] == self.kvs["AB"] == self.kvs["BA"] == self.kvs["BB"]:
            self.cnt = 1
        else:
            self.s = self.s[0] + self.kvs[self.s] + self.s[1]
            self.helper(self.s, 0)

    def helper(self, s, index):
        if len(s) == self.n:
            if s not in self.strs:
                self.cnt += 1
                self.strs.append(s)
            return
        for i in range(len(s) - 1):
            self.helper(s[:i+1] + self.kvs[s[i:i+2]] + s[i+1:], i)
    
    def get_cnt(self):
        return self.cnt

# n = int(input())
# aa, ab, ba, bb = input(), input(), input(), input()
# sys.setrecursionlimit(1000000)
# solver = Solver(n, aa, ab, ba, bb)
# solver.solve()
# print(solver.get_cnt()%(10**9 + 7))
import sys
inputs = [int(num) for num in sys.stdin.readline().strip().split(' ')]
print(inputs)
i = 0
res = []
while i < len(inputs):
    
    n = inputs[i]
    if i + n + 1 >= len(inputs):
        nums = inputs[i + 1:]
    else:
        nums = inputs[i + 1 : i + n + 1]
    res += sorted(list(set(nums)))
    i += n + 1
print(res)
# for r in res:
    # print(r)
     

