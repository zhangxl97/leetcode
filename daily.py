from typing import List
from tabulate import tabulate
from tqdm import tqdm
from singly_linked_list import ListNode, connect_nodes, print_nodes


class Solution:
    # 
    def kthLargestValue(self, matrix: List[List[int]], k: int) -> int:
        import heapq
        rows = len(matrix)
        cols = len(matrix[0])

        res = []

        xors = [[ 0 for _ in range(cols)] for _ in range(rows)]
        xors[0][0] = matrix[0][0]

        heapq.heappush(res, xors[0][0])

        for col in range(1, cols):
            xors[0][col] = xors[0][col - 1] ^ matrix[0][col]
            heapq.heappush(res, xors[0][col])
            if len(res) > k:
                heapq.heappop(res)
        
        # print(res)
        for row in range(1, rows):
            xors[row][0] = xors[row - 1][0] ^ matrix[row][0]
            heapq.heappush(res, xors[row][0])
            if len(res) > k:
                heapq.heappop(res)

        # print(res)
        for row in range(1, rows):
            for col in range(1, cols):
                xors[row][col] = xors[row - 1][col] ^ xors[row][col - 1] ^ xors[row - 1][col - 1] ^ matrix[row][col]
                heapq.heappush(res, xors[row][col])
                if len(res) > k:
                    heapq.heappop(res)
        
        print(tabulate(matrix))
        print(tabulate(xors))
        print(res)
        return heapq.heappop(res)

    # 692. 前K个高频单词
    def topKFrequent(self, words: List[str], k: int) -> List[str]:
        num = len(words)
        if num == 0:
            return []
        elif num == 1:
            return words

        from collections import Counter

        words = Counter(words)
        words = sorted(words.items(), key=lambda x:(-x[1], x[0]))

        res = []

        for i in range(k):
            res.append(words[i][0])
        return res

    # 810. 黑板异或游戏
    def xorGame(self, nums: List[int]) -> bool:
        if len(nums) % 2 == 0:
            return True
        
        from functools import reduce

        xor = reduce(lambda x, y : x ^ y, nums)
        return xor == 0

    # 1707. 与数组中元素的最大异或值
    def maximizeXor(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        nums.sort()
        from bisect import bisect_left
        res = []
        print(nums)
        print(queries)
        for xi, mi in queries:
            index = bisect_left(nums, mi)
            if index < len(nums) and nums[index] == mi:
                index += 1
            print(xi, mi, index)
            if index == 0:
                res.append(-1)
            else:
                tmp = 0
                for num in nums[:index]:
                    tmp = max(tmp, num ^ xi)
                res.append(tmp)
        return res

    # 664. 奇怪的打印机
    def strangePrinter(self, s: str) -> int:
        dp = [[0] * len(s) for _ in range(len(s))]
        # dp[0][0] = 1
        
        for i in range(len(s) - 1, -1, -1):
            dp[i][i] = 1
            for j in range(i + 1, len(s)):
                if s[i] == s[j]:
                    dp[i][j] = dp[i][j - 1]
                else:
                    tmp = float('inf')
                    for k in range(i, j):
                        tmp = min(tmp, dp[i][k] + dp[k+1][j])
                    dp[i][j] = tmp

        return dp[0][-1]

    # 1787. 使所有区间的异或结果为零
    # https://leetcode-cn.com/problems/make-the-xor-of-all-segments-equal-to-zero/solution/shi-suo-you-qu-jian-de-yi-huo-jie-guo-we-uds2/
    def minChanges(self, nums: List[int], k: int) -> int:
        
        from collections import Counter
        # from copy import deepcopy

        dp = [float('inf')] * (2 ** 10)
        dp[0] = 0
        size = len(nums)

        for i in range(k):

            cnt = 0
            count = Counter()

            for j in range(i, size, k):
                count[nums[j]] += 1
                cnt += 1
            
            # O(2^20 * N) TLE
            # dp_tmp = deepcopy(dp)
            # for mask in range(2 ** 10):
            #     tmp = float('inf')
            #     for x in range(2 ** 10):
            #         tmp = min(tmp, dp[mask ^ x] + cnt - count[x])
            #     dp_tmp[mask] = tmp
            # dp = dp_tmp
            
            # 优化
            t2min = min(dp)

            dp_tmp = [t2min] * (2 ** 10)

            for mask in range(2 ** 10):
                for x, countx in count.items():
                    dp_tmp[mask] = min(dp_tmp[mask], dp[mask ^ x] - countx)
                
            dp = [val + cnt for val in dp_tmp]

        
        return dp[0]

    # 1190. 反转每对括号间的子串
    def reverseParentheses(self, s: str) -> str:
        stack = []
        for i, c in enumerate(s):
            if c == "(":
                stack.append(i)
            elif c == ")":
                j = stack.pop()
                s = s[:j] + " " + s[j + 1:i][::-1] + " " + s[i + 1: ]
        s = s.replace(" ", "")
        return s

    # 560. 和为K的子数组
    def subarraySum(self, nums: List[int], k: int) -> int:
        from collections import Counter
        pre = 0
        counter = Counter()
        counter[0] = 1
        cnt = 0
        for num in nums:
            pre = pre + num
            cnt += counter[pre - k]
            counter[pre] += 1
        return cnt
        # res = 0
        # print(pre_sum)
        # for i in range(size):
        #     for j in range(i, size):
        #         # pre[i]−pre[j−1]==k --> pre[j−1]==pre[i]−k
        #         if pre_sum[j] - (pre_sum[i - 1] if i > 0 else 0) == k:
        #             res += 1
        # return res

    # 1074. 元素和为目标值的子矩阵数量
    def numSubmatrixSumTarget(self, matrix: List[List[int]], target: int) -> int:
        from collections import Counter
        def sub_array_sum(nums, k):

            size = len(nums)
            pre = 0
            counter = Counter()
            counter[0] = 1
            cnt = 0
            for num in nums:
                pre = pre + num
                cnt += counter[pre - k]
                counter[pre] += 1
            return cnt
        
        rows = len(matrix)
        cols = len(matrix[0])
        res = 0
        for row in range(rows):
            local = [0] * cols
            for r in range(row, rows):
                for c in range(cols):
                    local[c] += matrix[r][c]
                # print(local)
                res += sub_array_sum(local, target)
        
        return res

    # 2021.6
    # 前缀和，背包问题
    # 1744. 你能在你最喜欢的那天吃到你最喜欢的糖果吗？
    # [favoriteDay_i +1, (favoriteDay_i + 1) × dailyCap_i]
    def canEat(self, candiesCount: List[int], queries: List[List[int]]) -> List[bool]:
        # print(candiesCount)
        for i in range(1, len(candiesCount)):
            candiesCount[i] = candiesCount[i - 1] + candiesCount[i]
        # print(candiesCount)
        res = [False] * len(queries)

        for i in range(len(queries)):
            favorate_type, favorate_day, daily_cap = queries[i]
            # if favorate_day < candiesCount[favorate_type] and (candiesCount[favorate_type - 1] if favorate_type > 0 else 1) <= favorate_day * daily_cap:
            #     res[i] = True
            min_eat = favorate_day + 1
            max_eat = (favorate_day + 1) * daily_cap
            min_candy = 1 if favorate_type == 0 else candiesCount[favorate_type - 1] + 1
            max_candy = candiesCount[favorate_type]

            res[i] = min_eat <= max_candy and max_eat >= min_candy



        return res

    # 523. 连续的子数组和
    # 同余: 当 prefixSums[q]−prefixSums[p] 为 k 的倍数时，prefixSums[p] 和 prefixSums[q] 除以 k 的余数相同
    def checkSubarraySum(self, nums: List[int], k: int) -> bool:
        size = len(nums)
        if size < 2:
            return False
        
        table = {0:-1}
        pres = [0] * (size + 1)
        
        for i in range(size):
            pres[i + 1] = pres[i] + nums[i]
            mod = pres[i + 1] % k
            if table.get(mod) is not None:
                if i - table[mod] > 1:
                    return True
            else:
                table[mod] = i
        print(pres, table)
        
        return False

        
        # O(N^2) TLE
        # for i in range(size):
        #     for j in range(i + 2, size + 1):
        #         if ((pres[j] - pres[i]) / k) % 1 == 0:
        #             return True
        
        # return False

    # 525. 连续数组
    def findMaxLength(self, nums: List[int]) -> int:
        # size = len(nums)
        # if size == 1:
        #     return 1

        # pres = [0] * (size + 1)
        # for i in range(1, size):
        #     pres[i] = (1 if nums[i - 1] == 1 else -1) + pres[i - 1]
        
        # res = 0
        # for i in range(size):
        #     for j in range(i + 1, size + 1):
        #         if pres[j] - pres[i] == 0:
        #             res = max(res, j - i)
        # return res

        pre = 0

        table = {0:-1}

        res = 0

        for i in range(len(nums)):

            num = nums[i]
            pre += (num if num == 1 else -1)
            if table.get(pre) is None:
                table[pre] = i
            else:
                res = max(res, i - table[pre])
        
        return res

    # 203. 移除链表元素
    def removeElements(self, head: ListNode, val: int) -> ListNode:
        res = ListNode(-1)
        res.next = head

        pre = res

        p = head

        while p:
            if p.val == val:
                pre.next = p.next
            else:
                pre = pre.next
            
            p = p.next

        return res.next

    # 1049. 最后一块石头的重量 II
    # knapsack 问题
    # 对于该问题，定义二维布尔数组 dp，其中 dp[i+1][j] 表示前 i 个石头能否凑出重量 j 。特别地，dp[0][] 为不选任何石头的状态，因此除了 dp[0][0] 为真，其余 dp[0][j] 全为假。

    def lastStoneWeightII(self, stones: List[int]) -> int:
        size = len(stones)
        m = sum(stones) // 2

        # dp = [[False] * (m + 1) for _ in range(size + 1)]
        # dp[0][0] = True
        # for i in range(1, size + 1):
        #     for j in range(m + 1):
        #         if j >= stones[i - 1]:
        #             dp[i][j] = dp[i - 1][j] or dp[i - 1][j - stones[i - 1]]
        #         else:
        #             dp[i][j] = dp[i - 1][j]
                
        # for j in range(m, -1, -1):
        #     if dp[size][j]:
        #         return sum(stones) - 2 * j

        dp = [False] * (m + 1)
        dp[0] = True
        for i in range(size):
            for j in range(m, stones[i] - 1, -1):
                dp[j] = dp[j] or dp[j - stones[i]]

        for j in range(m, -1, -1):
            if dp[j]:
                return sum(stones) - 2 * j

    # 879. 盈利计划
    # 定义一个三维数组 dp 作为动态规划的状态，其中 dp[i][j][k] 表示在前 i 个工作中选择了 j 个员工，并且满足工作利润至少为 k 的情况下的盈利计划的总数目。
    # 可以发现 dp[i][j][k] 仅与 dp[i−1][..][..] 有关，所以本题可以用二维动态规划解决。
    def profitableSchemes(self, n: int, minProfit: int, group: List[int], profit: List[int]) -> int:
        # MOD = 10**9 + 7
        
        # length = len(group)
        # dp = [[[0] * (minProfit + 1) for _ in range(n + 1)] for _ in range(length + 1)]
        # dp[0][0][0] = 1

        # for i in range(1, length + 1):
        #     members, earn = group[i - 1], profit[i - 1]
        #     for j in range(n + 1):
        #         for k in range(minProfit + 1):
        #             if j < members:
        #                 dp[i][j][k] = dp[i - 1][j][k]
        #             else:
        #                 dp[i][j][k] = (dp[i - 1][j][k] + dp[i - 1][j - members][max(0, k - earn)]) % MOD
        
        # total = sum(dp[length][j][minProfit] for j in range(n + 1))
        # return total % MOD

        MOD = 10**9 + 7
        dp = [[0] * (minProfit + 1) for _ in range(n + 1)]
        for i in range(0, n + 1):
            dp[i][0] = 1
        for earn, members in zip(profit, group):
            for j in range(n, members - 1, -1):
                for k in range(minProfit, -1, -1):
                    dp[j][k] = (dp[j][k] + dp[j - members][max(0, k - earn)]) % MOD
        return dp[n][minProfit]

    # 1449. 数位成本和为目标值的最大数, Hard
    def largestNumber(self, cost: List[int], target: int) -> str:
        dp = [float("-inf")] * (target + 1)
        dp[0] = 0

        for c in cost:
            for j in range(c, target + 1):
                dp[j] = max(dp[j], dp[j - c] + 1)
                
        if dp[target] < 0:
            return "0"
        
        ans = list()
        j = target
        for i in range(8, -1, -1):
            c = cost[i]
            while j >= c and dp[j] == dp[j - c] + 1:
                ans.append(str(i + 1))
                j -= c

        return "".join(ans)

    # 852. 山脉数组的峰顶索引
    def peakIndexInMountainArray(self, arr: List[int]) -> int:
        left, right = 0, len(arr) - 1
        while left <= right:
            mid = (left + right) >> 1
            if 0 < mid < len(arr) - 1:
                if arr[mid - 1] <= arr[mid] >= arr[mid + 1]:
                    return mid
                elif arr[mid] >= arr[mid + 1]:
                    right = mid - 1
                else:
                    left = mid + 1
            elif mid == 0:
                left = mid + 1
            elif mid == len(arr) - 1:
                right = mid - 1
    
    # 877. 石子游戏
    def stoneGame(self, piles: List[int]) -> bool:
        alex = 0
        li = 0
        flag = True
        while piles != []:
            if flag:
                if len(piles) == 1:
                    alex += piles.pop()
                else:
                    if piles[0] >= piles[-1]:
                        alex += piles.pop(0)
                    else:
                        alex += piles.pop()
            else:
                if len(piles) == 1:
                    li += piles.pop()
                else:
                    if piles[0] >= piles[-1]:
                        li += piles.pop(0)
                    else:
                        li += piles.pop()
        return alex > li

    # 65. 有效数字
    def isNumber(self, s: str) -> bool:
        try:
            if(s == "inf" or s == "-inf" or s == "+inf" or s == "Infinity" or 
               s == "-Infinity" or s == "+Infinity"): return False 
            float(s)
            return True
        except:
            return False

    # 483. 最小好进制
    def smallestGoodBase(self, n: str) -> str:
        num = int(n)
        for m in range(num.bit_length(),2,-1):
            x = int(num**(1/(m-1)))
            if num == (x**m-1)//(x-1):
                return str(x)
        return str(num-1)
        # num = int(n)
        # def check(x, m):
        #     ans = 0
        #     for _ in range(m+1):
        #         ans = ans*x + 1
        #     return ans
        # ans = float("inf")
        # for i in range(1, 60):
        #     l = 2
        #     r = num
        #     while l < r:
        #         mid = l + (r - l)//2
        #         tmp = check(mid, i)
        #         if tmp == num:
        #             ans = min(ans, mid)
        #             break
        #         elif tmp < num:
        #             l = mid + 1
        #         else:
        #             r = mid
        # return str(ans)

    # 1239. 串联字符串的最大长度
    def maxLength(self, arr: List[str]) -> int:
        def dfs(arr, idx, flag, flags, length):
            if self.res == self.all:
                return
            self.res = max(self.res, length)

            if idx == len(arr):
                return 
            
            for i in range(idx, len(arr)):
                if flags[i] & flag == 0:
                    dfs(arr, i + 1, flag | flags[i], flags, length + len(arr[i]))

        self.res = 0

        flags = [0] * len(arr)
        for i, word in enumerate(arr):
            tmp = 0
            flag = True
            for c in word:
                if tmp & 1 << ord(c) - ord('a') != 0:
                    flag = False
                    break
                tmp |= 1 << ord(c) - ord('a')
            if flag is True:
                flags[i] = tmp
            else:
                arr[i] = ""
        
        self.all = sum([len(x) for x in arr])
        # print([bin (x) for x in flags])
        dfs(arr, 0, 0, flags, 0)
        return self.res


def main():
    s = Solution()

    # print(s.kthLargestValue([[10,9,5],[2,0,4],[1,0,9],[3,4,8]], 10))
    # print(s.topKFrequent(["the", "day", "is", "sunny", "the", "the", "the", "sunny", "is", "is"], k = 4))
    # print(s.maximizeXor([5,2,4,6,6,3], [[12,4],[8,1],[6,3]]))

    # print(s.strangePrinter("abcabc"))

    # print(s.minChanges(nums = [5,21,9,12,4,12,19,7,27,11,18,23,15,10,27,30,11,3], k = 8))
    
    # print(s.reverseParentheses(s = "a(bcdefghijkl(mno)p)q"))

    # print(s.subarraySum(nums = [1,2,3], k = 3))
    
    # print(s.numSubmatrixSumTarget(matrix = [[904]], target = 0))

    # print(s.canEat(candiesCount = [7,4,5,3,8], queries = [[0,2,2],[4,2,4],[2,13,1000000000]]))

    # print(s.checkSubarraySum(nums = [1,2,3], k = 5))

    # print(s.findMaxLength( nums = [0]))

    # nodes = connect_nodes([7,1,7,7])
    # print_nodes(nodes)
    # print_nodes(s.removeElements(nodes, 7))

    # print(s.lastStoneWeightII(stones = [31,26,33,21,40]))

    # print(s.profitableSchemes(64,0,[80, 40],[88, 88]))

    # print(s.largestNumber(cost = [2,2,2,2,2,2,2,2,2], target = 9))
    
    # print(s.peakIndexInMountainArray([3,5,3,2,0]))

    # print(s.stoneGame([5,3,4,5]))

    # print([s.isNumber(c) for c in ["-123456e789987987", "1a", "1e", "e3", "99e2.5", "--6", "-+3", "95a54e53"]])

    # print(s.smallestGoodBase("15"))

    print(s.maxLength(arr =["cha","r","act","ers"]))

if __name__ == "__main__":
    main()
