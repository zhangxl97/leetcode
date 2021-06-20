from typing import List
from tabulate import tabulate

class Solution:
    def isUnique(self, s):
        if s == "":
            return True

        kv = {}
        for c in s:
            if kv.get(ord(c)) is None:
                kv[ord(c)] = 1
            else:
                kv[ord(c)] += 1
                if kv[ord(c)] > 1:
                    return False
        return True

    def maxSub(self, s):
        if s == "" or len(s) == 1:
            return ""

        start_point = 0
        num_base = 0
        ans = 0
        curr = 0
        for i, c in enumerate(s):
            if num_base == 0:
                base = c
                num_base = 1
                continue

            if c == base:
                num_base += 1
            else:
                num_base -= 1
                curr += 2
                if num_base == 0:
                    ans += curr
                    curr = 0
                else:
                    if i < len(s) - 1 and s[i+1] == base:
                        if curr > ans:
                            ans = curr
                            start_point = i - curr + 1
                        curr = 0
                        num_base = 0
        return ans, s[start_point : start_point+ans]

    # 64
    def minPathSum(self, grid: List[List[int]]) -> int:
        nrows = len(grid)
        ncols = len(grid[0])

        for row in range(nrows):
            for col in range(ncols):
                if row == 0 and col == 0:
                    continue
                elif row == 0 and col > 0:
                    grid[row][col] = grid[row][col - 1] + grid[row][col]
                elif col == 0 and row > 0:
                    grid[row][col] = grid[row - 1][col] + grid[row][col]
                else:
                    grid[row][col] = min(grid[row - 1][col], grid[row][col - 1]) + grid[row][col]
        
        return grid[-1][-1]

    # 62
    def uniquePaths(self, m: int, n: int) -> int:
        dp = [[1 for _ in range(n)] for _ in range(m)]
        for row in range(1, m):
            for col in range(1, n):
                dp[row][col] = dp[row - 1][col] + dp[row][col - 1]
        return dp[-1][-1]

    # 413
    def numberOfArithmeticSlices(self, nums: List[int]) -> int:
        size = len(nums)
        if size < 3:
            return 0

        dp = [0 for _ in range(size)]

        for i in range(2, size):
            if nums[i] - nums[i - 1] == nums[i - 1] - nums[i - 2]:
                dp[i] = dp[i - 1] + 1
        return sum(dp)




    # 343
    def integerBreak(self, n: int) -> int:
        if n <= 3:
            return n - 1
        elif n == 4:
            return n
        dp = [0] * (n + 1)
        for i in range(1, 5):
            dp[i] = i

        for i in range(5, n + 1):
            max_pro = 0
            start = 1
            while start <= i // 2:
                left = i - start
                curr_pro = dp[start] * dp[left]
                if curr_pro > max_pro:
                    dp[i] = curr_pro
                    max_pro = curr_pro
                start += 1
        return dp[-1]

    # 279
    def numSquares(self, n: int) -> int:
        if n <= 3:
            return n
        from math import sqrt
        def is_square(num):
            return int(sqrt(num)) ** 2 == num

        if is_square(n):
            return 1

        dp = [0] * (n + 1)
        for i in range(1, 4):
            dp[i] = i
        
        for i in range(4, n + 1):
            if is_square(i):
                dp[i] = 1
            else:
                min_num = float('inf')
                value = int(sqrt(i))
                while value >= 1:
                    left = i - value * value
                    curr_num = dp[value * value] + dp[left]
                    min_num = min(min_num, curr_num)
                    value -= 1
                dp[i] = min_num
        return dp[-1]

    # 91
    def numDecodings(self, s: str) -> int:
        if s.startswith('0'):  # 开头有 ‘0’ 直接返回
            return 0

        n = len(s)
        dp = [1] * (n+1)  # 重点是 dp[0], dp[1] = 1, 1

        for i in range(2, n+1):
            if s[i-1] == '0' and s[i-2] not in '12':  # 出现前导 ‘0’ 的情况，不能解码，直接返回
                return 0
            if s[i-2:i] in ['10', '20']:  # 只有组合在一起才能解码
                dp[i] = dp[i-2]
            elif '10' < s[i-2:i] <= '26': # 两种解码方式
                dp[i] = dp[i-1] + dp[i-2]
            else:                         # '01'到 ‘09’ 或 > '26'，只有单独才能解码
                dp[i] = dp[i-1]
        return dp[n]

    # 300
    def lengthOfLIS(self, nums: List[int]) -> int:
        # O(N^2)
        # size = len(nums)
        # dp = [0] * size
        # res = 0

        # for i in range(size):
        #     max_i = 1
        #     for j in range(i):
        #         if nums[i] > nums[j]:
        #             max_i = max(max_i, dp[j] + 1)

        #     dp[i] = max_i
        #     res = max(res, dp[i])
        # return res

        # O(NlogN)
        def binary_search(tails, len_tail, num):
            left, right = 0, len_tail - 1
            while left <= right:
                mid = (left + right) // 2
                if tails[mid] == num:
                    return mid
                elif tails[mid] < num:
                    left = mid + 1
                else:
                    right = mid - 1
            return left

        size = len(nums)
        tails = [0] * size
        len_tail = 0
        for num in nums:
            index = binary_search(tails, len_tail, num)
            tails[index] = num
            if index == len_tail:
                len_tail += 1
        return len_tail

    # 646 
    def findLongestChain(self, pairs: List[List[int]]) -> int:
        pairs.sort()
        size = len(pairs)
        dp = [0] * size 
        res = 0
        for i, pair in enumerate(pairs):
            max_len = 1
            for j in range(i):
                if pair[0] > pairs[j][1]:
                    max_len = max(max_len, dp[j] + 1)
            dp[i] = max_len
        print(pairs)
        print(dp)
            # res = max(res, dp[i])
        return max(dp)

    # 376
    def wiggleMaxLength(self, nums: List[int]) -> int:
        size = len(nums)
        if size == 1:
            return 1
        up = 1
        down = 1
        for i in range(1, size):
            if nums[i] - nums[i - 1] > 0:
                up = down + 1
            elif nums[i] - nums[i - 1] < 0:
                down = up + 1
        return max(up, down)

    # 1143
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        size1 = len(text1)
        size2 = len(text2)
        dp = [[0 for _ in range(size1 + 1)] for _ in range(size2 + 1)]
        for index2 in range(1, size2 + 1):
            for index1 in range(1, size1 + 1):
                if text1[index1 - 1] == text2[index2 - 1]:
                    dp[index2][index1] = dp[index2 - 1][index1 - 1] + 1
                else:
                    dp[index2][index1] = max(dp[index2 - 1][index1], dp[index2][index1 - 1])
        print(tabulate(dp))
        return dp[-1][-1]

    # 0/1 knapsack
    # 容量为V，物品个数为N，各个价值为values[i], 体积为weights[i]
    # dp[i][v] 表示前i件物品放进一个使用容量为v的背包可以获得的最大价值
    # dp[i][v] = max(dp[i-1][v],dp[i-1][v-weights[i]]+values[i])
    # dp[j] = max(dp[j], dp[w - w[i]] + c[i])
    # 返回为能得到的最大价值
    def knapsack_01(self, values: List[int], weights: List[int], W: int, N: int):
        # T=O(WN) S=O(WN)
        dp = [[0 for _ in range(W + 1)] for _ in range(N + 1)]

        for i in range(1, N + 1):
            weight = weights[i - 1]
            value = values[i - 1]
            for w in range(1, W + 1):
                if w >= weight:
                    dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - weight] + value)
                else:
                    dp[i][w] = dp[i - 1][w]

        j = W
        for i in range(N, -1, -1):
            if j >= weights[i - 1]:
                if dp[i][j] == dp[i - 1][j - weights[i - 1]] + values[i - 1]:
                    j -= weights[i - 1]
                    print("i=",i)
            
        return dp[-1][-1]

        # S=O(W)
        # dp = [0 for _ in range(W + 1)]
        # path = [[0 for _ in range(W + 1)] for _ in range(N + 1)]

        # for i in range(1, N + 1):
        #     weight = weights[i - 1]
        #     value = values[i - 1]
        #     for w in range(W, weight - 1, -1):
        #         tmp = dp[w-weight]+value
        #         if tmp > dp[w]:
        #             dp[w] = tmp
        #             path[i][w] = 1
        #         # dp[w] = max(dp[w], dp[w-weight]+value)
        
        #     # print(dp)
        # print(tabulate(path))
        # j = W
        # for i in range(N, -1, -1):
        #     if path[i][j] == 1:
        #         print("i=:",i)
        #         j -= weights[i - 1]
        # return dp[-1]

    # 完全背包问题: 即取的物品数量不限
    # https://www.bilibili.com/video/BV1C7411K79w?p=2&t=1082
    def knapsack_full(self, values: List[int], weights: List[int], W: int, N: int):
        # O(N^3)
        # 逆序
        # dp = [0 for _ in range(W + 1)]
        # dp = [0] * (W + 1)
        # for i in range(N):
        #     weight = weights[i]
        #     value = values[i]
        #     # 利用上一条数据，旧值
        #     for w in range(W, 0, -1):
        #         for k in range(w//weight + 1):  # w // weight 当前w容量下能取值为weight的物品最大个数
        #             dp[w] = max(dp[w], dp[w - k * weight] + k * value)
        #     # print(dp)
        # return dp[-1]
        dp = [[0] * (W + 1) for _ in range(N + 1)]
        for i in range(1,N+1,1):
            for j in range(W+1):
                if j < weights[i-1]:
                    dp[i][j] = dp[i-1][j]
                else:
                    dp[i][j] = max(dp[i-1][j],dp[i][j-weights[i-1]]+values[i-1])
        print(tabulate(dp))
        j=W
        for i in range(N,0,-1):
            while (j-weights[i-1]) >=0:
                if ((dp[i][j-weights[i-1]]+values[i-1])>dp[i-1][j]):
                    j-=weights[i-1]
                    print('i=',i)#记录选择的物品
                else:
                    break
        return dp[-1][-1]
        # O(N^2)
        # 正序
        # 利用新值
        # dp[j] = max(dp[j], dp[j-w[i]] + c[i])
        # dp = [0] * (W + 1) 

        # for i in range(N):
        #     for j in range(weights[i], W + 1):
        #         dp[j] = max(dp[j], dp[j - weights[i]] + values[i])

        # return dp[-1]

    # 416
    # 可以看成一个背包大小为 sum/2 的 0-1 背包问题。
    # 返回为是否存在这个背包，初始为True
    def canPartition(self, nums: List[int]) -> bool:
        target = sum(nums)
        if target % 2 != 0:
            return False
        target = target // 2

        dp = [False] * (target + 1)
        dp[0] = True

        for num in nums:
            for w in range(target, num - 1, -1):
                dp[w] = dp[w] or dp[w - num]
        # print(dp)
        return dp[-1]

    # 494 
    # P符号前位1，N符号位为-1
    # sum(P) - sum(N) = target
    # sum(P) - sum(N) + sum(P) + sum(N) = target + sum(P) + sum(N)
    # sum(P) = (target + sum(nums)) / 2
    # 返回为选取的组合数，初始为1
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        sum_num = sum(nums)
        if sum_num < target or (sum_num + target) % 2 != 0:
            return 0
        
        W = (sum_num + target) // 2
        dp = [0] * (W + 1)
        dp[0] = 1

        for num in nums:
            for w in range(W, num - 1, -1):
                dp[w] = dp[w] + dp[w - num]

        return dp[-1]

    # 474
    # m: 0, n: 1
    # 多维费用的 0-1 背包问题，有两个背包大小，0 的数量和 1 的数量。
    # 返回为选取物品数量，不需要初始
    def findMaxForm(self, strs: List[str], m: int, n: int) -> int:
        from collections import Counter
        strs_ = [Counter(s) for s in strs]
        
        dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]

        for s in strs_:
            zero_c = s['0']
            one_c = s['1']
            for zero in range(m, zero_c - 1, -1):
                for one in range(n, one_c - 1, -1):
                    dp[zero][one] = max(dp[zero][one], dp[zero-zero_c][one-one_c]+1)


        print(dp)
        return dp[-1][-1]

    # 322
    # 因为硬币可以重复使用，因此这是一个完全背包问题。
    # 返回取的物品的个数，不需要初始
    def coinChange(self, coins: List[int], amount: int) -> int:
        if amount == 0:
            return 0

        dp = [0 for _ in range(amount + 1)]

        for coin in coins:
            for j in range(coin, amount + 1):
                if j == coin:
                    dp[j] = 1
                elif dp[j] == 0 and dp[j - coin] != 0:
                    dp[j] = dp[j - coin] + 1
                elif dp[j - coin] != 0:
                    dp[j] = min(dp[j], dp[j - coin] + 1)
                

        return dp[-1] if dp[-1] else -1

    # 518
    # 硬币组合数，返回物品选取组合可能总数，初始为1
    def change(self, amount: int, coins: List[int]) -> int:
        if amount == 0:
            return 1
        
        dp = [0] * (amount + 1)
        dp[0] = 1

        for coin in coins:
            for j in range(coin, amount + 1):
                dp[j] = dp[j] + dp[j - coin]
        return dp[-1]

    # 139
    # 该问题涉及到字典中单词的使用顺序，也就是说物品必须按一定顺序放入背包中，例如下面的 dict 就不够组成字符串 "leetcode"：["lee", "tc", "cod"]
    # 求解顺序的完全背包问题时，对物品的迭代应该放在最里层，对背包的迭代放在外层，只有这样才能让物品按一定顺序放入背包中。
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:

        dp = [False] * (len(s) + 1)
        dp[0] = True

        for j in range(len(s) + 1):
            for word in wordDict:
                size = len(word)
                if j >= size:
                    dp[j] = dp[j] or (dp[j - size] and s[j - size: j] == word)
            print(dp)

        return dp[-1]

    # 377
    # 请注意，顺序不同的序列被视作不同的组合。
    def combinationSum4(self, nums: List[int], target: int) -> int:
        dp = [0] * (target + 1)
        dp[0] = 1

        for j in range(target + 1):
            for num in nums:
                if j >= num:
                    dp[j] = dp[j] + dp[j-num]

        return dp[-1]

    # 309. 最佳买卖股票时机含冷冻期
    # profit[i][0]: 持有
    # profit[i][1]: 不持有+冷冻期内
    # profit[i][2]: 不持有+不在冷冻期内
    # 不限次
    def maxProfit(self, prices: List[int]) -> int:
        if len(prices) <= 1:
            return 0

        profits_0, profits_1, profits_2 = -prices[0], 0, 0

        for price in prices[1:]:
            new_0 = max(profits_0, profits_2 - price)
            new_1 = profits_0 + price
            new_2 = max(profits_1, profits_2)

            profits_0 = new_0
            profits_1 = new_1
            profits_2 = new_2
        
        return max(profits_1, profits_2)
    
    # 714. 买卖股票的最佳时机含手续费，Medium
    # 不限次
    def maxProfit_fee(self, prices: List[int], fee: int) -> int:
        if len(prices) <= 1:
            return 0

        profit_0 = -prices[0]  # 持有
        profit_1 = 0    # 不持有

        for price in prices[1:]:
            new_0 = max(profit_0, profit_1 - price)
            new_1 = max(profit_1, profit_0 + price - fee)

            profit_0 = new_0
            profit_1 = new_1

        return profit_1

    # 123. 买卖股票的最佳时机 III, Hard
    # 只能交易k次
    def maxProfit_k_trans(self, k: int, prices: List[int]) -> int:
        days = len(prices)
        if days <= 1:
            return 0
        # print(prices)
        dp = [[0 for _ in range(days)] for i in range(k + 1)]  # 交易0，1，2，...,k次的总收益

        for i in range(1, k + 1):
            max_profit = -prices[0]
            for j in range(1, days):
                dp[i][j] = max(dp[i][j - 1], prices[j] + max_profit)  # max(不操作，当前日期卖出的最大总收益)
                max_profit = max(dp[i-1][j] - prices[j], max_profit)  # max(如果上一次交易不交易，且当前日期买入，最大收益)
                # print(max_profit)

                # print(tabulate(dp))
        return dp[-1][-1]

    # 375, 猜数字大小
    def getMoneyAmount(self, n: int) -> int:
        dp = [[0] * (n+1) for _ in range(n+1)]
        for i in range(1,n):
            dp[i][i+1] = i
        for low in range(n-1, 0 ,-1):
            for high in range(low+1, n+1):
                dp[low][high] = min(x + max(dp[low][x-1], dp[x+1][high]) for x in range(low,high))
                print(low, high)
                print(tabulate(dp))
        return dp[1][n]

# 303
# Your NumArray object will be instantiated and called as such:
# obj = NumArray(nums)
# param_1 = obj.sumRange(left,right)
class NumArray:
    # 求区间 i ~ j 的和，可以转换为 sum[j + 1] - sum[i]，其中 sum[i] 为 0 ~ i - 1 的和。
    def __init__(self, nums: List[int]):
        self.sums = nums
        for i in range(1, len(nums)):
            self.sums[i] = self.sums[i - 1] + nums[i]
        print(self.sums)

    def sumRange(self, left: int, right: int) -> int:
        return self.sums[right] - (self.sums[left - 1] if left > 0 else 0)


def main():
    s = Solution()

    # print(s.isUnique("abcdef"))
    # print(s.maxSub("aaaaabbbabab"))
    # print(s.minPathSum(grid = [[1],[1],[4]]))
    # print(s.uniquePaths(m=3,n=7))

    # s = NumArray([-2, 0, 3, -5, 2, -1])
    # print(s.sumRange(0,2))
    # print(s.sumRange(2,5))
    # print(s.sumRange(0,5))

    # print(s.numberOfArithmeticSlices([1,2,3,4,5]))
    # print(s.integerBreak(8))
    # print(s.numSquares(13))
    # print(s.numDecodings("10011"))
    # print(s.lengthOfLIS(nums = [10,9,2,5,3,7,101,18]))
    # print(s.findLongestChain([[3,4],[2,3],[1,2]]))
    # print(s.wiggleMaxLength([1,7,4,9,2,5]))
    # print(s.longestCommonSubsequence("bsbininm", "jmjkbkjkv"))

    print(s.knapsack_01(values=[1,3,5,9],weights=[1,2,3,100],W=10,N=4))
    # print(s.canPartition([2,4,4]))
    # print(s.findTargetSumWays([1,1,1,1,1],3))
    # print(s.findMaxForm(strs = ["10", "0001", "111001", "1", "0"], m = 5, n = 3))
    # print(s.coinChange([2,5,10,1],27))

    # print(s.change(amount = 10, coins = [10]))
    # print(s.wordBreak("applepenapple", ["apple","pen"]))
    # print(s.combinationSum4(nums = [1,2,3], target = 4))

    # print(s.maxProfit([1,2,3,0,2]))
    # print(s.maxProfit_k_trans(k = 2, prices = [3,2,6,5,0,3]))

    # print(s.getMoneyAmount(n = 6))

if __name__ == "__main__":
    main()
