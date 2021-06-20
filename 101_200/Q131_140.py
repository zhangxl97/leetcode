from typing import List


class Solution:
    # 131. 分割回文串, Medium
    def partition(self, s: str) -> List[List[str]]:

        def dfs(s, idx, tmp, results, is_pal):
            if idx == len(s):
                results.append(tmp)
            
            for i in range(idx, len(s)):
                if is_pal[idx][i]:
                    dfs(s, i + 1, tmp+[s[idx:i + 1]], results, is_pal)

        size = len(s)
        is_pal = [[True] * size for _ in range(size)]
        for i in range(size - 1, -1, -1):
            for j in range(i + 1, size):
                is_pal[i][j] = is_pal[i + 1][j - 1] and (s[i] == s[j])
        print(is_pal)
        results = []
        dfs(s, 0, [], results, is_pal)
        return results

    # 132. 分割回文串 II, Hard
    def minCut(self,  s: str) -> int:
        # dfs, TLE
        # def dfs(s, idx, tmp, is_pal):
        #     if idx == len(s):
        #         self.num = min(self.num, len(tmp) - 1)
        #         return 

        #     # print(s)
        #     for i in range(idx, len(s)):
        #         if is_pal[idx][i]:
        #             dfs(s, i + 1, tmp + [s[idx:i+1]], is_pal)

        size = len(s)
        is_pal = [[True] * size for _ in range(size)]
        for i in range(size - 1, -1, -1):
            for j in range(i + 1, size):
                is_pal[i][j] = is_pal[i + 1][j - 1] and (s[i] == s[j])
        
        # self.num = float('inf')
        # dfs(s, 0, [], is_pal)
        # return self.num

        # 设 f[i] 表示字符串的前缀 s[0:i+1] 的最少分割次数。要想得出 f[i] 的值，我们可以考虑枚举 s[0:i+1] 分割出的最后一个回文串，这样我们就可以写出状态转移方程：
        # f[i] = \min_{0 \leq j < i} \{ f[j] \} + 1, \quad 其中 ~ s[j+1:i+1] ~是一个回文串
        # 即我们枚举最后一个回文串的起始位置 j+1，保证 s[j+1:i+1] 是一个回文串，
        # 那么 f[i] 就可以从 f[j] 转移而来，附加 1 次额外的分割次数。

        dp = [float('inf')] * size

        for i in range(size):
            if is_pal[0][i]:
                dp[i] = 0
            else:
                for j in range(i):
                    if is_pal[j+1][i]:
                        dp[i] = min(dp[i], dp[j] + 1)
        return dp[-1]

            




def main():
    s = Solution()

    print(s.minCut("aab"))

if __name__ == "__main__":
    main()

