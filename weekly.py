from collections import Counter
from itertools import takewhile
from typing import List
from numpy.core.einsumfunc import _parse_possible_contraction
from tabulate import tabulate

class Solution:
    # 5479. Thousand Separator
    def thousandSeparator(self, n: int) -> str:
        s = list(str(n))
        n = len(s)
        i = n - 1
        while i > 2:
            s.insert(i - 2, '.')
            i -= 3
        return ''.join(s)

    # 5480. Minimum Number of Vertices to Reach All Nodes
    def get_val(self, x: List, i: int, has: list, full: List, dicts: dict):
        if i > len(x) - 1:
            return
        com = set(has + dicts[i])
        if com == set(full):
            # 符合条件记录
            x[i] = 1
            print(x)
            x[i] = 0
        x[i] = 1
        self.get_val(x, i + 1, list(com), full, dicts)
        # 当前位置不取 执行一次
        x[i] = 0
        self.get_val(x, i + 1, has, full, dicts)

    def findSmallestSetOfVertices(self, n: int, edges: List[List[int]]) -> List[int]:
        dicts = {i: [] for i in range(n)}
        for fr, to in edges:
            dicts[fr].append(to)

        full = [i for i in range(n)]
        temp = dicts[0]
        for i in range(1, n):
            temp += dicts[i]
        return list(set(full) - set(temp))

    # 5481. Minimum Numbers of Function Calls to Make Target Array
    # res = all ones in binary + (the largest number of bits - 1)
    def minOperations(self, nums: List[int]) -> int:
        return sum(bin(x).count('1') for x in nums) + len(bin(max(nums))) - 3   # bin(x) = 'obxxxxx', so -2 -1 = -3

    # 5726
    def arraySign(self, nums: List[int]) -> int:
        cnt_lower_0 = 0
        for num in nums:
            if num == 0:
                return 0
            elif num < 0:
                cnt_lower_0 += 1
        if cnt_lower_0 % 2:
            return -1
        else:
            return 1

    # 5727
    def findTheWinner(self, n: int, k: int) -> int:
        def find_first_not_None(candidate):
            start = 0
            start = 0
            while candidate[start] is None:
                start += 1
            return start
        candidate = [i+1 for i in range(n)]

        size = n
        start = 0

        while size > 1:
            step = 0
            step_num = (k % size - 1) if k % size != 0 else size - 1
            while step < step_num:
                start += 1
                if start == n:
                    start = find_first_not_None(candidate)
                if candidate[start] is not None:
                    step += 1

            candidate[start] = None
            print(start)
            
            start += 1
            if start == n:
                start = find_first_not_None(candidate)
            while candidate[start] is None:
                start += 1
                if start == n:
                    start = find_first_not_None(candidate)
            size -= 1
            print(candidate)
        return candidate[start]

    # 5728
    def minSideJumps(self, obstacles: List[int]) -> int:
        size = len(obstacles)
        dp = [[0 for _ in range(size)] for _ in range(3)]
        dp[0][0] = 1
        dp[2][0] = 1

        for i, obstacle in enumerate(obstacles):
            if obstacle:
                dp[obstacle - 1][i] = float('inf')

        for point in range(size):
            road = 0
            for road in range(3):
                if dp[road][point] == float('inf'):
                    continue
                else:
                    if road == 0 and dp[road][point - 1] == float('inf') and dp[road + 1][point] == float('inf')  and flag is True:
                        flag = False
                    else:
                        if dp[road - 1][point] == 0:
                            dp[road][point] = min
                        dp[road][point] = min(dp[road][point - 1], (dp[road - 1][point] + 1) if dp[road - 1][point] != 0 else float('inf'), dp[(road + 1) if road < 2 else 0][point] + 1)
                        if road == 0 and flag is False:
                            flag = True

            if flag:
                point += 1
        return dp[-1][-1]
        print(tabulate(dp))

    # 5730
    def replaceDigits(self, s: str) -> str:

        for i in range(1, len(s), 2):
            s = s[:i] + chr(ord(s[i-1]) + int(s[i])) + s[i + 1:]

        return s
    
    # 5731
    def maximumElementAfterDecrementingAndRearranging(self, arr: List[int]) -> int:
        arr.sort()
        for i in range(1, len(arr)):
            if arr[i] - arr[i - 1] > 1:
                arr[i] = arr[i - 1] + 1
        return arr[-1]

    # 5732 ?
    def closestRoom(self, rooms: List[List[int]], queries: List[List[int]]) -> List[int]:

        import numpy as np
        rooms = np.array(rooms)
        room_sizes = rooms[:,1]
        room_ids = rooms[:,0]
        ans = [-1] * len(queries)

        for i in range(len(queries)):
            preferred, min_size = queries[i]
            tmp = room_sizes - min_size
            tmp = tmp >= 0
            tmp = tmp.nonzero()[0]

            if tmp.size == 0:
                continue
            
            indexes = room_ids[tmp]
            bias = np.abs(indexes - preferred)
            ans[i] = indexes[np.argmin(bias)].tolist()

        return ans

    # 5746
    def getMinDistance(self, nums: List[int], target: int, start: int) -> int:

        if nums[start] == target:
            return 0
        
        left = start - 1
        right = start + 1
        ans = float('inf')
        while left >= 0 or right < len(nums):
            left_target = left if left >= 0 and nums[left] == target else None
            right_target = right if right < len(nums) and nums[right] == target else None

            if left_target and right_target:
                ans = min(ans, right_target - start)
            elif left_target is None and right_target is not None:
                ans = min(ans, right_target - start)
            elif right_target is None and left_target is not None:
                ans = min(ans, start - left_target)
            left -= 1
            right += 1

        return ans
    
    # 5747 将字符串拆分为递减的连续值
    def splitString(self, s: str) -> bool:
        def dfs(sub_s, past):
            if sub_s == "":
                return True
            
            flag = False
            for i in range(len(sub_s)):
                if int(sub_s[:i+1]) == past - 1:
                    flag = flag or dfs(sub_s[i+1:], int(sub_s[:i+1]))
            return flag

        # ans = False
        for i in range(len(s) - 1):
            # print(int(s[:i+1]))
            ans = dfs(s[i+1:], int(s[:i+1]))
            if ans:
                return True
        return False
            
        
    # 5749 ?
    def getMinSwaps(self, num: str, k: int) -> int:

        def next_permu(nums):
            left_i = len(nums) - 2
            right_i = len(nums) - 1

            # e.g 2 3 1
            # left_i: 1, right_i: 2
            while left_i >= 0 and int(nums[left_i]) >= int(nums[left_i + 1]):
                left_i -= 1
            # left_i: 0
            if left_i >= 0:
                # 2 3 1
                while int(nums[right_i]) <= int(nums[left_i]):
                    right_i -= 1
                nums[left_i], nums[right_i] = nums[right_i], nums[left_i]
            nums[left_i + 1:] = sorted(nums[left_i + 1:])
        
        def pop_sort(nums):
            ans = 0
            for i in range(len(num) - 1):
                for j in range(len(nums) - i - 1):
                    if nums[j] > nums[j + 1]:
                        nums[j], nums[j + 1] = nums[j + 1], nums[j]
                        ans += 1
            return ans

        base = num
        num = list(num)
        step = 0
        while step < k:
            next_permu(num)
            if int(''.join(num)) > int(base):
                step += 1
        print(num)

        base = list(base)
        print(base)

        i = 0
        ans = 0
        while i < len(base):
            if base[i] == num[i]:
                i += 1
                continue
            else:
                j = i
                while base[j] != num[i]:
                    j += 1
                ans += j - i
                base = base[:i] + [base[j]] + base[i:j] + base[j+1:]
                i += 1

        return ans

    # 5748 ?
    def minInterval(self, intervals: List[List[int]], queries: List[int]) -> List[int]:
        # TLE
        ans = [-1] * len(queries)
        for i in range(len(intervals)):
            intervals[i].append(intervals[i][1] - intervals[i][0] + 1)
        intervals.sort(key=lambda x:(x[2], x[0], x[1]))    
        # import bisect 
        for index, num in enumerate(queries):
            for i in range(len(intervals)):
                if intervals[i][0] <= num <= intervals[i][1]:
                    ans[index] = intervals[i][2]
                    break
        return ans

    # 5750. 人口最多的年份
    def maximumPopulation(self, logs: List[List[int]]) -> int:
        num = len(logs)
        if num == 1:
            return logs[0][0]
        elif num == 0:
            return 0

        min_year = float('inf')
        max_year = 0
        for log in logs:
            birth, death = log
            min_year = min(min_year, birth)
            max_year = max(max_year, death)
        print(min_year, max_year)
        ans_year = None
        people = 0

        years = [0] * (max_year - min_year)

        for log in logs:
            birth, death = log
            for i in range(birth, death):
                years[i - min_year] += 1
            
        return years.index(max(years)) + min_year

    # 5751. 下标对中的最大距离
    def maxDistance(self, nums1: List[int], nums2: List[int]) -> int:
        i, j = 0, 0
        len1, len2 = len(nums1), len(nums2)

        ans = 0

        while i < len1 and j < len2:
            if i <= j and nums1[i] <= nums2[j]:
                ans = max(ans, j - i)
                j += 1
            elif i > j:
                j += 1
            else:
                i += 1
        
        return ans

    # 5752. 子数组最小乘积的最大值 ?前缀和 + 单调栈
    def maxSumMinProduct(self, nums: List[int]) -> int:
        print(nums)
        size = len(nums)
        if size == 1:
            return nums[0] * nums[0]

        # for i in range(size):
        #     for j in range(i, size):
        #         # print(nums[i:j+1], max(nums[i:j+1]), sum(nums[i:j+1]))
        #         if min(nums[i:j+1]) * sum(nums[i:j+1]) > res:
        #             print(nums[i:j+1])
        #         res = max(res, min(nums[i:j+1]) * sum(nums[i:j+1]))
        # return res % (10**9 + 7)

        # res = max(min(nums) * sum(nums), self.maxSumMinProduct(nums[1:]), self.maxSumMinProduct(nums[:-1]))
        # return res % (10**9 + 7)

        pre_sum = [nums[0]]
        for num in nums[1:]:
            pre_sum.append(pre_sum[-1] + num)
        
        right_first_smaller = [None] * size
        left_first_smaller = [None] * size

        stack = []

        for i in range(size):
            while stack and nums[i] < nums[stack[-1]]:
                right_first_smaller[stack.pop()] = i
            stack.append(i)
        
        stack = []
        for i in range(size - 1, -1, -1):
            while stack and nums[i] < nums[stack[-1]]:
                left_first_smaller[stack.pop()] = i
            stack.append(i)

        print(right_first_smaller)
        print(left_first_smaller)
        
        res = 0
        for i in range(size):
            left = left_first_smaller[i]
            right = right_first_smaller[i]
            res = max(res, nums[i] * ((pre_sum[right] if right is not None else pre_sum[-1]) - (pre_sum[left - 1] if left is not None and left > 0 else 0)))
        
        return res % (10**9 + 7)

    # 5753. 有向图中最大颜色值 ? 拓扑排序 + DP
    def largestPathValue(self, colors: str, edges: List[List[int]]) -> int:
        num = len(colors)
        degree = [0] * num
        dp = [[0 for _ in range(26)] for _ in range(num)]
        connection = {}

        # 计算每个顶点的入度 以及 连接
        for start, end in edges:
            degree[end] += 1
            if connection.get(start) is None:
                connection[start] = [end]
            else:
                connection[start].append(end)

        # 将入度为0的顶点入队，并将初始出现的字母个数置1
        queue = []
        for i in range(num):
            if degree[i] == 0:
                queue.append(i)
                dp[i][ord(colors[i]) - 97] = 1

        
        # 按顺序出队，对于从点x到点y，即更新点y时候每个字符出现的最多次数
        # 并将该顶点删除（意思是将所有与该点相连的边都删掉，即将边另一端对应的点的入度减1)，若删除该点后与该点相连的点入度变为了0，则将该点加入队列。
        ans = 1
        cnt = 0
        while queue != []:
            node = queue.pop(0)
            cnt += 1
            if connection.get(node) is not None:
                for next_node in connection[node]:
                    for j in range(26):
                        dp[next_node][j] = max(dp[next_node][j], dp[node][j] + (ord(colors[next_node]) - 97 == j))
                        ans = max(ans, dp[next_node][j])
                    
                    degree[next_node] -= 1
                    if degree[next_node] == 0:
                        queue.append(next_node)
        # print(cnt)

        # 因为只有入度为 0 的点才能入队，故若存在环，环上的点一定无法入队。
        # 所以只需统计入过队的点数之和是否等于点的总数 n 即可。
        if cnt != num:
            return -1

        return ans

    # 5742. 将句子排序
    def sortSentence(self, s: str) -> str:
        ans = [""] * 9
        start = 0
        for i, c in enumerate(s):
            if '1' <= c <= '9':
                ans[int(c) - 1] = s[start:i]
                start = i + 1
            if c == " ":
                start = i + 1
        ans = " ".join(ans).strip()
        return ans

    # 5743. 增长的内存泄露
    def memLeak(self, memory1: int, memory2: int) -> List[int]:

        require = 1

        while True:
            if memory1 < require and memory2 < require:
                return [require, memory1, memory2]
            
            elif memory1 >= memory2:
                memory1 -= require
            else:
                memory2 -=require
            
            require += 1

    # 5744. 旋转盒子
    def rotateTheBox(self, box: List[List[str]]) -> List[List[str]]:
        import numpy as np
        box = np.array(box)
        print(box)
        box = np.rot90(box, k=3).tolist()
        
        print(tabulate(box))
        rows = len(box)
        cols = len(box[0])
        for row in range(rows - 2, -1, -1):
            for col in range(cols):
                if box[row][col] == '#':
                    i = row + 1
                    pre = row
                    while i < rows and box[i][col] == '.':
                        box[pre][col] = '.'
                        box[i][col] = '#'
                        pre = i
                        i += 1

        # box.rotate(90)
        print(tabulate(box))
        return box

    # 5212. 向下取整数对和
    def sumOfFlooredPairs(self, nums: List[int]) -> int:

        # 转化为数组操作，仍然 TLE
        # import numpy as np
        # size = len(nums)

        # nums_1 = np.matrix(nums)
        # nums_1 = nums_1.reshape(-1,1)
        # # nums_1 = np.tile(nums_1, (1, size))

        # nums_2 = np.matrix(nums)
        # nums_2 = nums_2.reshape(1,-1)
        # # nums_2 = np.tile(nums_2, (size, 1))


        # print(nums_1)
        # print(nums_2)

        # ans = nums_1 // nums_2
        # return np.sum(ans).tolist() % (10 ** 9 + 7)


        # O(N^2) 优化重复元素
        # nums.sort()
        nums.sort()
        size = len(nums)
        repeats = [1] * size
        i = size - 1

        while i >= 0:
            cnt = 1
            j = i - 1
            while j >= 0 and nums[j] == nums[i]:
                cnt += 1
                j -= 1
            repeats[i] = cnt
            i -= cnt
        
        i = size - 1
        ans = 0
        while i >= 0:
            curr = nums[i]
            tmp = 1
            for num in nums[:i]:
                tmp += curr // num
            ans += tmp * repeats[i]
            i -= repeats[i]
        
        return ans

        # O(N^2)
        # ans = size
        # for i in range(1, size):
        #     for j in range(i):
        #         ans += nums[i] // nums[j]
        # return ans

    # 5759. 找出所有子集的异或总和再求和
    def subsetXORSum(self, nums: List[int]) -> int:
        from functools import reduce
        def dfs(nums, idx, tmp, k):
            if len(tmp) == k:
                self.res += reduce(lambda x, y : x ^ y, tmp)
                return 
            

            for i in range(idx, len(nums)):
                dfs(nums, i+1, tmp+[nums[i]], k)
        
        self.res = 0
        for i in range(1, len(nums)+1):
            dfs(nums, 0, [], i)
        
        return self.res

    # 5760. 构成交替字符串需要的最小交换次数
    def minSwaps(self, s: str) -> int:
        # from collections import Counter
        # nums = Counter(s)
        # if abs(nums['0'] - nums['1']) > 1:
        #     return -1
        
        s00, s01, s10, s11 = 0, 0, 0, 0
        for i, c in enumerate(s):
            if i % 2 == 0:
                if c == '1':
                    s00 += 1
                else:
                    s11 += 1
            else:
                if c == '1':
                    s10 += 1
                else:
                    s01 += 1
        if s00 != s01 and s10 != s11:
            return -1
        if s00 == s01 and s10 != s11:
            return s00
        if s00 != s01 and s10 == s11:
            return s10
        return min(s00, s10)



    # 5762. 恰有 K 根木棍可以看到的排列数目
    def rearrangeSticks(self, n: int, k: int) -> int:
        # import itertools
        # cnt = 0
        # for sticks in itertools.permutations(list(range(1, n + 1)),n):
        #     curr_max = sticks[0]
        #     see = 1
        #     for stick in sticks[1:]:
        #         if stick > curr_max:
        #             see += 1
        #             curr_max = stick
        #     if see == k:
        #         cnt += 1
        # return cnt

        # dp = [[0 for _ in range(k + 1)] for _ in range(n + 1)]
        # dp[0][0] = 1
        
        # for i in range(1, n + 1):
        #     for j in range(1, min(k, i) + 1):
        #         if i == j:
        #             dp[i][j] = 1
        #         else:
        #             dp[i][j] = dp[i - 1][j] * (i - 1) + dp[i - 1][j - 1] 
        # # print(tabulate(dp))
        # return dp[-1][-1] % (10 ** 9 + 7)

        dp = [0 for _ in range(k + 1)]
        dp[1] = 1
        print(dp)
        for i in range(2, n + 1):
            # g = [0] * (k + 1)
            for j in range(min(k, i), 0, -1):
                dp[j] = (dp[j] * (i - 1) + dp[j - 1]) % (10 ** 9 + 7)
            # dp = g
            print(dp)
        # print(tabulate(dp))
        return dp[-1] 


    # 5763. 哪种连续子字符串更长
    def checkZeroOnes(self, s: str) -> bool:
        maxs = {'1':0, '0':0}

        i = 0
        while i < len(s):
            base = s[i]
            j = i + 1
            while j < len(s) and s[j] == base:
                j += 1
            
            maxs[base] = max(maxs[base], j - i)

            i = j
        
        return maxs['1'] > maxs['0']

    # 5764. 准时到达的列车最小时速
    def minSpeedOnTime(self, dist: List[int], hour: float) -> int:
        
        def check(dist, hour, speed):
            time = hour
            for i, dis in enumerate(dist):
                if i == len(dist) - 1:
                    time -= dis / speed
                else:
                    time -= ceil(dis / speed)
            if time >= 0:
                return True
            return False

        if hour <= len(dist) - 1:
            return -1

        from math import ceil
        left, right = 1, 10000000

        while left < right:
            mid = (left + right) >> 1
            if check(dist, hour, mid):
                right = mid
            else:
                left = mid + 1

        return left

    # 5765. 跳跃游戏 VII
    def canReach(self, s: str, minJump: int, maxJump: int) -> bool:
        from functools import reduce 
        if s[-1] == '1':
            return False
        dp = [False] * len(s)

        dp[0] = True

        for i in range(1, len(s)):
            if s[i] == '0':
                # print(max(0, i - maxJump), max(0, i - minJump))
                # dp[i] = reduce(lambda x,y:x or y, dp[max(0, i - maxJump) : max(0, i - minJump) + 1])
                for j in range(max(0, i - minJump), max(0, i - maxJump) - 1, -1):
                    if dp[j]:
                        dp[i] = True
                        break
        return dp[-1]

        # BFS, TLE
        # queue = [0]

        # while queue != []:
            
        #     size = len(queue)

        #     for _ in range(size):

        #         pos = queue.pop(0)
        #         if pos == len(s) - 1:
        #             return True

        #         for j in range(min(len(s) - 1, pos + maxJump), pos + minJump - 1, -1):
        #             if s[j] == '0':
        #                 queue.append(j)

        # return False

    # 5766. 石子游戏 VIII
    def stoneGameVIII(self, stones: List[int]) -> int:
        def max_sum(stones):
            s = stones[0] + stones[1]
            max_s = stones[0] + stones[1]
            j = 1
            for i in range(2, len(stones)):
                s += stones[i]
                if s > 0 or s > max_s:
                    max_s = s
                    j = i
            if j < len(stones) - 2 and stones[j + 1] < 0 and stones[j + 2] < 0:
                j += 1
                max_s += stones[j]
            return max_s, j


        alice, bob = 0, 0
        alice_turn = True
        print(stones)
        while len(stones) > 1:

            max_s, j = max_sum(stones)
            if alice_turn:
                alice += max_s
                alice_turn = not alice_turn
            else:
                bob += max_s
                alice_turn = not alice_turn
            stones = [max_s] + stones[j+1:]
            print(stones, alice, bob)
        return alice - bob


    # No.53
    # 5754. 长度为三且各字符不同的子字符串
    def countGoodSubstrings(self, s: str) -> int:
        size = len(s)
        if size <= 2:
            return 0
        
        res = 0
        for i in range(size - 2):
            if s[i] != s[i + 1] and s[i] != s[i + 2] and s[i + 1] != s[i + 2]:
                res += 1
        
        return res
    # 5755. 数组中最大数对和的最小值
    def minPairSum(self, nums: List[int]) -> int:
        # DFS, TLE
        # def dfs(nums, tmp):
        #     if nums == []:
        #         self.res = min(self.res, tmp)
        #         return 

        #     num0 = nums[0]
        #     nums = nums[1:]
        #     for i, num in enumerate(nums):
        #         dfs(nums[:i]+nums[i+1:], max(num0+num, tmp))
            
        # self.res = float('inf')
        # dfs(nums, 0)

        # return self.res
        nums.sort()
        res = 0
        for i in range(len(nums)//2):
            res = max(res, nums[i] + nums[len(nums) - i - 1])


        return res
    # 5757. 矩阵中最大的三个菱形和
    def getBiggestThree(self, grid: List[List[int]]) -> List[int]:
        def check(x, border):
            if x < 0 or x >= border:
                return False
            else:
                return True

        print(tabulate(grid))
        from heapq import heappush, heappop
        from math import ceil
        rows = len(grid)
        cols = len(grid[0])
        heap = []
        areas = {}

        heights = rows if rows % 2 == 1 else rows - 1

        for row in range(rows):
            for col in range(cols):
                # print("row, col: ", row, col)
                for height in range(1, heights + 1, 2):
                    local = grid[row][col]
                    cnt = 1
                    # print("height: ", height)
                    for h in range(1, height):
                        # print(h, height)
                        if h <= (height - 1) // 2:
                            bais = h
                        else:
                            bais = height - h - 1
                        # print("bais: ", bais)
                        if check(row + h, rows) and check(col - bais, cols) and check(col + bais, cols):
                            cnt += 1
                            if bais == 0:
                                local += grid[row + h][col]
                            else:
                                local += grid[row + h][col - bais] + grid[row + h][col + bais]
                        else:
                            break
                    if cnt == height:
                        # print(row, col, height, local)
                        if areas.get(local) is None:
                            areas[local] = True
                            heappush(heap, local)
                            if len(heap) > 3:
                                heappop(heap)
        heap.sort(reverse=True)
        return heap
    # 5756. 两个数组最小的异或值之和
    def minimumXORSum(self, nums1: List[int], nums2: List[int]) -> int:
        nums1.sort(reverse=True)
        nums2.sort(reverse=True)
        res = 0
        for num1 in nums1:
            tmp = float('inf')
            id2 = None
            for i, num2 in enumerate(nums2):
                if num1 ^ num2 < tmp:
                    id2 = i
                    tmp = num1 ^ num2
            res += tmp
            nums2 = nums2[:id2] + nums2[id2+1:]
        return res

    # No
    # 5772. 检查某单词是否等于两单词之和
    def isSumEqual(self, firstWord: str, secondWord: str, targetWord: str) -> bool:
        num1 = 0
        for c in firstWord:
            num1 = num1 * 10 + ord(c) - 97
        num2 = 0
        for c in secondWord:
            num2 = num2 * 10 + ord(c) - 97
        num3 = 0
        for c in targetWord:
            num3 = num3 * 10 + ord(c) - 97
        
        return num1 + num2 == num3
    # 5773. 插入后的最大值
    def maxValue(self, n: str, x: int) -> str:
        ori = n
        if n[0] == "-":
            for i in range(1, len(n)):
                if int(n[i]) > x:
                    n = n[:i] + str(x) + n[i:]
                    break
        else:
            for i in range(len(n)):
                if int(n[i]) <= x:
                    n = n[:i] + str(x) + n[i:]
                    break
        if n == ori:
            n = n + str(x)
        return n
    # 5774. 使用服务器处理任务
    def assignTasks(self, servers: List[int], tasks: List[int]) -> List[int]:
        from collections import OrderedDict
        ans = [-1] * len(tasks)

        servers = [(i, server) for i, server in enumerate(servers)]
        tasks = [(i, task) for i, task in enumerate(tasks)]

        servers.sort(key=lambda x : (x[1], x[0]))
        
        busy = OrderedDict()
        busy[servers[0][0]] = - tasks[0][1]
        
        for i in range(1, len(servers)):
            busy[servers[i][0]] = 0
        
        # print(busy, servers)
        ans[tasks[0][0]] = servers[0][0]
        tasks.pop(0)

        work = 0
        while tasks != []:
            if work < len(tasks):
                work += 1
            for id in busy:
                if busy[id] < 0:
                    busy[id] += 1
            # print(busy)
            finished = []
            for i in range(work):
                # print(i, len(tasks), work)
                task = tasks[i][1]
                for id in busy:
                    if busy[id] == 0:
                        busy[id] = - task
                        ans[tasks[i][0]] = id
                        finished.append(tasks[i])
                        break

            for f in finished:
                work -= 1
                tasks.remove(f)

        return ans

    # 5775. 准时抵达会议现场的最小跳过休息次数, Hard
    # DP。。。没想到啊没想到我们用 f[i][j] 表示经过了 dist[0] 到 dist[i−1] 的 i 段道路，并且跳过了 j 次的最短用时。
    def minSkips(self, dist: List[int], speed: int, hoursBefore: int) -> int:
        # 可忽略误差
        from math import ceil
        EPS = 1e-8
        
        n = len(dist)
        f = [[float("inf")] * (n + 1) for _ in range(n + 1)]
        f[0][0] = 0.
        for i in range(1, n + 1):
            for j in range(i + 1):
                if j != i:
                    f[i][j] = min(f[i][j], ceil(f[i - 1][j] + dist[i - 1] / speed - EPS))
                if j != 0:
                    f[i][j] = min(f[i][j], f[i - 1][j - 1] + dist[i - 1] / speed)
        
        for j in range(n + 1):
            if f[n][j] < hoursBefore + EPS:
                return j
        return -1

    # 5776. 判断矩阵经轮转后是否一致
    def findRotation(self, mat: List[List[int]], target: List[List[int]]) -> bool:
        import numpy as np
        mat = np.array(mat)
        target = np.array(target)
        for i in range(4):
            tmp = np.rot90(mat, k=i)
            if np.all(tmp == target):
                return True

        return False

    # 5777. 使数组元素相等的减少操作次数
    def reductionOperations(self, nums: List[int]) -> int:
        from collections import Counter
        nums = Counter(nums)

        keys = sorted(list(nums.keys()), reverse=True)
        pre_sum = [0] * len(keys)
        for i in range(len(keys) - 1):
            pre_sum[i + 1] = pre_sum[i] + nums[keys[i]]
        return sum(pre_sum)

    # 5779. 装包裹的最小浪费空间
    def minWastedSpace(self, packages: List[int], boxes: List[List[int]]) -> int:
        mod = 10 ** 9 + 7
        res = float('inf')
        for boxs in boxes:
            boxs.sort()
            waste = 0
            flag = True
            for package in packages:
                if package > boxs[-1]:
                    flag = False
                    break
                for box in boxs:
                    if package <= box:
                        waste += box - package
                        break
            if flag:
                res = min(res, waste)
        return res % mod if res != float('inf') else -1


    # No.54
    # 5767. 检查是否区域内所有整数都被覆盖
    def isCovered(self, ranges: List[List[int]], left: int, right: int) -> bool:
        size = len(ranges)
        if size == 1:
            return ranges[0][0] <= left and ranges[0][1] >= right
        
        ranges.sort()
        past_left, past_right = ranges[0][0], ranges[0][1]

        compose = []

        for l, r in ranges[1:]:
            if l > past_right + 1:
                compose.append([past_left, past_right])
                past_left, past_right = l, r
            elif r >= past_right:
                past_right = r

        compose.append([past_left, past_right])
        print(compose)
        for l, r in compose:
            if l <= left and r >= right:
                return True
        return False
    # 5768. 找到需要补充粉笔的学生编号
    def chalkReplacer(self, chalk: List[int], k: int) -> int:
        k = k % sum(chalk)
        n = len(chalk)
        i = 0
        while k >= chalk[i]:
            k -= chalk[i]
            i += 1
            if i == n:
                i = 0
            # print(k)
        return i
    # 5202. 最大的幻方
    def largestMagicSquare(self, grid: List[List[int]]) -> int:
        import numpy as np
        def check(grid, step) -> bool:
            sum_ = np.sum(grid, axis=0)
            base = sum_[0]
            if np.sum(sum_ == base) != step:
                return False
            sum_ = np.sum(grid, axis=1)
            if np.sum(sum_ == base) != step:
                return False
            diag_1, diag_2 = 0, 0
            for i in range(grid.shape[0]):
                diag_1 += grid[i][i]
                diag_2 += grid[i][step - i - 1]
            return diag_1 == base and diag_2 == base

        grid = np.array(grid)
        (rows, cols) = grid.shape
        max_n = min(rows, cols)
        
        for i in range(max_n, 1, -1):
            for start_i in range(rows - i + 1):
                for start_j in range(cols - i + 1):
                    if check(grid[start_i:start_i+i, start_j:start_j+i], i):
                        return i

        return 1
    # 5770. 反转表达式值的最少操作次数
    def minOperationsToFlip(self, expression: str) -> int:
        states = []
        ops = []
                
        for c in expression:
            if c in '01)':
                if c == '0':
                    states.append((0, 1))
                elif c == '1':
                    states.append((1, 0))
                else:
                    assert(ops[-1] == '(')
                    ops.pop()
                    
                if len(ops) > 0 and ops[-1] != '(':
                    op = ops.pop()
                    p2, q2 = states.pop()
                    p1, q1 = states.pop()
                    if op == '&':
                        states.append((min(p1, p2), min(q1 + q2, 1 + min(q1, q2))))
                    else:
                        states.append((min(p1 + p2, 1 + min(p1, p2)), min(q1, q2)))
            else:
                ops.append(c)
        
        return max(states[-1])

    # No.245
    # 5784. 重新分配字符使所有字符串都相等
    def makeEqual(self, words: List[str]) -> bool:
        size = len(words)
        if size == 1:
            return True
        from collections import Counter
        cnt = Counter()
        for word in words:
            tmp = Counter(word)
            for key in tmp:
                cnt[key] += tmp[key]
        print(cnt)
        for key in cnt:
            if cnt[key] % size != 0:
                return False
        return True
    # 5786. 可移除字符的最大数目
    # 二分
    def maximumRemovals(self, s: str, p: str, removable: List[int]) -> int:
        from copy import deepcopy
        def check(tmp_s, p):
            i, j = 0, 0
            while i < len(tmp_s) and j < len(p):
                if tmp_s[i] == p[j]:
                    i += 1
                    j += 1
                else:
                    i += 1
            return j == len(p)
        
        s = list(s)
        left, right = 0, len(removable) - 1
        
        while left <= right:
            mid = (left + right) >> 1
            tmp_s = deepcopy(s)
            for i in range(mid + 1):
                tmp_s[removable[i]] = " "
            if check(tmp_s, p):
                left = mid + 1
            else:
                right = mid - 1
        return left
    # 5785. 合并若干三元组以形成目标三元组
    def mergeTriplets(self, triplets: List[List[int]], target: List[int]) -> bool:
        x, y, z = 0, 0, 0
        for x1, y1, z1 in triplets:
            if x1 <= target[0] and y1 <= target[1] and z1 <= target[2]:
                x = max(x, x1)
                y = max(y, y1)
                z = max(z, z1)
        return x == target[0] and y == target[1] and z == target[2]

    #No246
    # 5788. 字符串中的最大奇数
    def largestOddNumber(self, num: str) -> str:
        i = len(num) - 1
        while i >= 0:
            if int(num[i]) % 2 == 1:
                return num[:i+1]
            i -= 1
        return ""
    # 5789. 你完成的完整对局数
    def numberOfRounds(self, startTime: str, finishTime: str) -> int:
        # 转化为分钟
        t0 = 60 * int(startTime[:2]) + int(startTime[3:])
        t1 = 60 * int(finishTime[:2]) + int(finishTime[3:])
        if t1 < t0:
            # 此时 finishTime 为第二天
            t1 += 1440
        # 第一个小于等于 finishTime 的完整对局的结束时间
        t1 = t1 // 15 * 15
        return max(0, (t1 - t0)) // 15

        # hour_start = int(startTime[:2])

        # min_start = int(startTime[3:])
        # hour_end = int(finishTime[:2])
        # min_end = int(finishTime[3:])

        # hour = hour_end - hour_start
        # flag = False
        # if hour < 0:
        #     hour += 24
        # if hour == 0 and min_end < min_start:
        #     flag = True
        #     hour += 23

        # if min_start == 0:
        #     min_start = 0
        # elif min_start <= 15:
        #     min_start = 15
        # elif min_start <= 30:
        #     min_start = 30
        # elif min_start <= 45:
        #     min_start = 45
        # else:
        #     min_start = 60
        
        # if min_end < 15:
        #     min_end = 0
        # elif min_end < 30:
        #     min_end = 15
        # elif min_end < 45:
        #     min_end = 30
        # elif min_end < 60:
        #     min_end = 45
        
        # if flag is False:
        #     if min_end < min_start and hour > 0:
        #         min_end += 60
        #         hour -= 1 
        #     return max(0, (min_end - min_start)) // 15 + hour * 4
        # else:
        #     return (min_end + 60 - min_start) // 15 + hour * 4
    # 5791. 统计子岛屿
    def countSubIslands(self, grid1: List[List[int]], grid2: List[List[int]]) -> int:
        def dfs(grid, row, col, rows, cols, directions):
            if row < 0 or col < 0 or row == rows or col == cols or grid[row][col] == 0:
                return 
            
            grid[row][col] = 0
            if grid1[row][col] == 0:
                self.flag = 0

            for d_row, d_col in directions:
                nr = row + d_row
                nc = col + d_col
                dfs(grid, nr, nc, rows, cols, directions)
        
        rows = len(grid2)
        cols = len(grid2[0])
        directions = [[1,0],[-1,0],[0,-1],[0,1]]
        cnt = 0
        
        for row in range(rows):
            for col in range(cols):
                self.flag = 1
                if grid2[row][col] == 1:
                    dfs(grid2, row, col, rows, cols, directions)
                    if self.flag == 1:
                        cnt += 1
        return cnt
    # 5790. 查询差绝对值的最小值
    # 前缀和 pre[i][c] 表示数组 nums 的前缀 a[0..i−1] 中包含元素 c 的个数
    def minDifference(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        # 元素 c 的最大值
        C = 100

        n = len(nums)
        # 前缀和
        pre = [[0] * (C + 1)]
        for i, num in enumerate(nums):
            pre.append(pre[-1][:])
            pre[-1][num] += 1

        ans = list()
        for left, right in queries:
            # last 记录上一个出现的元素
            # best 记录相邻两个元素差值的最小值
            last = 0
            best = float("inf")
            for j in range(1, C + 1):
                if pre[left][j] != pre[right + 1][j]:
                    if last != 0:
                        best = min(best, j - last)
                    last = j
            
            if best == float("inf"):
                best = -1
            ans.append(best)
        
        return ans






# 5729        
class MKAverage:

    def __init__(self, m: int, k: int):
        self.m = m
        self.k = k
        self.size = 0
        self.values = []

    def addElement(self, num: int) -> None:
        self.values.append(num)
        self.size += 1
        if self.size > self.m:
            self.values = self.values[-self.m:]
            self.size = self.m

    def calculateMKAverage(self) -> int:
        print(self.values)
        if self.size < self.m:
            return -1
        tmp = sorted(self.values)
        tmp = tmp[self.k: self.size - self.k]
        print(tmp)
        tmp_size = self.size - 2 * self.k
        res = int(sum(tmp) / tmp_size)
        return res

# 5731 
class SeatManager:

    def __init__(self, n: int):
        self.__seats = [i for i in range(1, n + 1)]


    def reserve(self) -> int:
        return self.__seats.pop(0)

    def unreserve(self, seatNumber: int) -> None:
        import bisect
        bisect.insort(self.__seats, seatNumber)

# 146
class LRUCache:

    def __init__(self, capacity: int):
        from collections import OrderedDict
        self.capacity = capacity
        self.elements = OrderedDict()

    def get(self, key: int) -> int:
        if self.elements.get(key) is not None:
            self.elements.move_to_end(key)
            return self.elements[key]
        else:
            return -1

    def put(self, key: int, value: int) -> None:
        if self.elements.get(key) is not None:
            self.elements.move_to_end(key)
        else:
            if len(self.elements) == self.capacity:
                self.elements.popitem(0)
        self.elements[key] = value

# 5761. 找出和为指定值的下标对
import numpy as np
class FindSumPairs:
    def __init__(self, nums1: List[int], nums2: List[int]):
        self.nums1 = nums1
        self.nums2 = nums2
        self.cnt = Counter(nums2)

    def add(self, index: int, val: int) -> None:
        self.cnt[self.nums2[index]] -= 1
        self.nums2[index] += val
        self.cnt[self.nums2[index]] += 1


    def count(self, tot: int) -> int:
        res = 0
        for num in self.nums1:
            res += self.cnt[tot - num]
        return res


# 432
class Node:
    pre = None
    next = None
    val = 0

class AllOne:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        # self.values = {}
        self.values = []

    def inc(self, key: str) -> None:
        """
        Inserts a new key <Key> with value 1. Or increments an existing key by 1.
        """
        # self.values.setdefault(key, 0)
        # self.values[key] += 1



    def dec(self, key: str) -> None:
        """
        Decrements an existing key by 1. If Key's value is 1, remove it from the data structure.
        """
        # if self.values[key] == 1:
        #     self.values.pop(key)
        # else:
        #     self.values[key] -= 1


    def getMaxKey(self) -> str:
        """
        Returns one of the keys with maximal value.
        """
        # return max(self.values.items(), key=lambda x:x[1], default=[""])[0]


    def getMinKey(self) -> str:
        """
        Returns one of the keys with Minimal value.
        """
        # return min(self.values.items(), key=lambda x:x[1], default=[""])[0]



# Your AllOne object will be instantiated and called as such:
# obj = AllOne()
# obj.inc(key)
# obj.dec(key)
# param_3 = obj.getMaxKey()
# param_4 = obj.getMinKey()

# 895. 最大频率栈
class FreqStack(object):

    def __init__(self):
        import collections
        # 记录每个key出现的次数
        self.freq = collections.Counter()
        # 记录每个出现次数的数字
        self.group = collections.defaultdict(list)
        # 最大频率
        self.maxfreq = 0

    def push(self, x):
        f = self.freq[x] + 1
        self.freq[x] = f
        if f > self.maxfreq:
            self.maxfreq = f
        self.group[f].append(x)

        print(self.freq, self.group, self.maxfreq)


    def pop(self):
        x = self.group[self.maxfreq].pop()
        self.freq[x] -= 1
        if not self.group[self.maxfreq]:
            self.maxfreq -= 1

        print(self.freq, self.group, self.maxfreq)
        return x


def main():
    s = Solution()

    # 5479
    # print(s.thousandSeparator(123456712312312389))
    # 5480
    # print(s.findSmallestSetOfVertices(n = 5, edges = [[0,1],[2,1],[3,1],[1,4],[2,4]]))
    # 5481
    # print(s.minOperations(nums = [4,2,5]))
    # print(s.arraySign(nums = [-1,1,-1,1,-1]))
    # print(s.findTheWinner(n=6,k=5))
    # print(s.minSideJumps(obstacles = [0,1,2,3,0]))

    # obj = MKAverage(6, 1)
    # obj.addElement(3)
    # obj.addElement(1)
    # obj.addElement(12)
                            
                            
    # obj.addElement(5)
    # obj.addElement(3)
    # obj.addElement(4)
    # print(obj.calculateMKAverage())

    # print(s.replaceDigits(s = "a1c1e1"))
    # seatManager = SeatManager(5)
    # print(seatManager.reserve())
    # print(seatManager.reserve())
    # seatManager.unreserve(2)
    # print(seatManager.reserve())
    # print(seatManager.reserve())
    # print(seatManager.reserve())
    # print(seatManager.reserve())
    # seatManager.unreserve(5)

    # print(s.closestRoom(rooms = [[2,2],[1,2],[3,2]], queries = [[3,1],[3,3],[5,2]]))
    # print(s.getMinDistance([5,3,6], 5, 2))
    # print(s.getMinSwaps(num = "00123", k = 4))
    # print(s.minInterval(intervals = [[2,3],[2,5],[1,8],[20,25]], queries = [2,19,5,22]))
    # print(s.splitString("00900089"))

    # lRUCache = LRUCache(2)
    # lRUCache.put(1, 0)
    # lRUCache.put(2, 2)
    # print(lRUCache.get(1))
    # lRUCache.put(3, 3)
    # print(lRUCache.get(2))
    # lRUCache.put(4, 4)
    # print(lRUCache.get(1))
    # print(lRUCache.get(3))
    # print(lRUCache.get(4))

    # print(s.maximumPopulation([[2033,2034],[2039,2047],[1998,2042],[2047,2048],[2025,2029],[2005,2044],[1990,1992],[1952,1956],[1984,2014]]))

    # print(s.maxDistance(nums1 = [5,4], nums2 = [3,2]))
    # print(s.maxSumMinProduct(nums = [3,1,5,6,4,2]))
    # print(s.largestPathValue(colors = "abaca", edges = [[0,1],[0,2],[2,3],[3,4]]))


    # print(s.sortSentence("is2 sentence4 This1 a3"))
    # print(s.memLeak(memory1 = 8, memory2 = 11))
    # print(s.rotateTheBox(box = [["#","#","*",".","*","."],
    #         ["#","#","#","*",".","."],
    #         ["#","#","#",".","#","."]]))
    # print(s.sumOfFlooredPairs(nums = [7,7,7,7,7,7,7]))
    # print(s.subsetXORSum(nums = [5,1,6]))
    # print(s.minSwaps("00011110110110000000000110110101011101111011111101010010010000000000000001101101010010001011110000001101111111110000110101101101001011000011111011101101100110011111110001100110001110000000001100010111110100111001001111100001000110101111010011001"))

    # findSumPairs = FindSumPairs([1, 1, 2, 2, 2, 3], [1, 4, 5, 2, 5, 4])
    # print(findSumPairs.count(7))
    # findSumPairs.add(3, 2)
    # print(findSumPairs.count(8))
    # print(findSumPairs.count(4))
    # findSumPairs.add(0, 1)
    # findSumPairs.add(1, 1)
    # print(findSumPairs.count(7))

    # print(s.rearrangeSticks(n = 3, k = 2))


    # freq = FreqStack()
    # freq.push(5)
    # freq.push(7)
    # freq.push(5)
    # freq.push(7)
    # freq.push(4)
    # freq.push(5)
    # print(freq.pop())
    # print(freq.pop())
    # print(freq.pop())
    # print(freq.pop())

    # print(s.checkZeroOnes("1101000101111"))
    # print(s.minSpeedOnTime(dist = [1,3,2], hour = 2.7))
    # print(s.canReach("011010", 2, 5))
    # print(s.stoneGameVIII(stones = [-39,-23,-43,-7,25,-36,-32,17,-42,-5,-11]))

    # 53
    # print(s.minPairSum(nums = [3,5,4,2,4,6]))
    # print(s.getBiggestThree([[15,14,15,19,6,18,15,14],[18,7,8,10,3,5,11,19],[20,11,10,1,6,3,16,3],[7,14,4,9,18,14,13,3],[20,5,15,3,9,8,16,16],[6,7,4,12,2,19,11,20],[20,11,10,3,4,9,5,15],[13,10,4,18,16,2,4,20]]))
    # print(s.minimumXORSum([100,26,12,62,3,49,55,77,97], [98,0,89,57,34,92,29,75,13]))

    #
    # print(s.isSumEqual(firstWord = "aaa", secondWord = "a", targetWord = "aaaa"))
    # print(s.maxValue("-132", 3))
#     print(s.assignTasks([338,890,301,532,284,930,426,616,919,267,571,140,716,859,980,469,628,490,195,664,925,652,503,301,917,563,82,947,910,451,366,190,253,516,503,721,889,964,506,914,986,718,520,328,341,765,922,139,911,578,86,435,824,321,942,215,147,985,619,865],
# [773,537,46,317,233,34,712,625,336,221,145,227,194,693,981,861,317,308,400,2,391,12,626,265,710,792,620,416,267,611,875,361,494,128,133,157,638,632,2,158,428,284,847,431,94,782,888,44,117,489,222,932,494,948,405,44,185,587,738,164,356,783,276,547,605,609,930,847,39,579,768,59,976,790,612,196,865,149,975,28,653,417,539,131,220,325,252,160,761,226,629,317,185,42,713,142,130,695,944,40,700,122,992,33,30,136,773,124,203,384,910,214,536,767,859,478,96,172,398,146,713,80,235,176,876,983,363,646,166,928,232,699,504,612,918,406,42,931,647,795,139,933,746,51,63,359,303,752,799,836,50,854,161,87,346,507,468,651,32,717,279,139,851,178,934,233,876,797,701,505,878,731,468,884,87,921,782,788,803,994,67,905,309,2,85,200,368,672,995,128,734,157,157,814,327,31,556,394,47,53,755,721,159,843]))
    # print(s.minSkips([1,1,1,1,1], 10000000, 1))

    # print(s.findRotation(mat = [[0,0,0],[0,1,0],[1,1,1]], target = [[1,1,1],[0,1,0],[0,0,0]]))
    # print(s.reductionOperations(nums = [1,1,2,2,3]))
    # print(s.minWastedSpace(packages = [2,3,5], boxes = [[4,8],[2,8]]))

    # print(s.isCovered([[37,49],[5,17],[8,32]],29,49))
    # print(s.chalkReplacer(chalk = [3,4,1,2], k = 25))
    # print(s.largestMagicSquare(grid = [[7,1,4,5,6],[2,5,1,6,4],[1,5,4,3,2],[1,2,7,3,4]]))

    # print(s.makeEqual(['a','b']))
    # print(s.maximumRemovals("abcbddddd", "abcd", [3,2,1,4,5,6]))
    # print(s.mergeTriplets(triplets = [[1,3,4],[2,5,8]], target = [2,5,8]))

    # print(s.largestOddNumber(num = "35427"))
    # print(s.numberOfRounds(startTime = "23:48", finishTime = "23:16"))
    # print(s.countSubIslands(grid1 = [[1,0,1,0,1],[1,1,1,1,1],[0,0,0,0,0],[1,1,1,1,1],[1,0,1,0,1]], grid2 = [[0,0,0,0,0],[1,1,1,1,1],[0,1,0,1,0],[0,1,0,1,0],[1,0,0,0,1]]))
    print(s.minDifference([43,43,43,43,43,43,43,43,43,43,43,43,43,43,43,43,43,43,43,43,43,43,43,43,43,43,43,43,43,43,43,43,43,43,43,43,43,43,43,43,43,43,43,43,43,43,43,43,43,43,43,43,43,43,43,43,43,43,43,43,43,43,43,43,43,43,43,43,43,43,43,43,43,43,43,43,43,43,43,43,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,94,94,94,94,94,94,94,94,94,94,94,94,94,94,94,94,94,94,94,94,94,94,94,94,94,94,94,94,94,94,94,94,94,94,94,94,94,94,94,94,94,94,94,94,94,94,94,94,94,94,94,94,94,94,94,94,94,94,94,94,94,94,94,94,94,94,94,94,94,94,94,94,94,94,94,94,94,94,94,94,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,53,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,42,42,42,42,42,42,42,42,42,42,42,42,42,42,42,42,42,42,42,42,42,42,42,42,42,42,42,42,42,42,42,42,42,42,42,42,42,42,42,42,42,42,42,42,42,42,42,42,42,42,42,42,42,42,42,42,42,42,42,42,42,42,42,42,42,42,42,42,42,42,42,42,42,42,42,42,42,42,42,42,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,52,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,89,89,89,89,89,89,89,89,89,89,89,89,89,89,89,89,89,89,89,89,89,89,89,89,89,89,89,89,89,89,89,89,89,89,89,89,89,89,89,89,89,89,89,89,89,89,89,89,89,89,89,89,89,89,89,89,89,89,89,89,89,89,89,89,89,89,89,89,89,89,89,89,89,89,89,89,89,89,89,89,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,86,86,86,86,86,86,86,86,86,86,86,86,86,86,86,86,86,86,86,86,86,86,86,86,86,86,86,86,86,86,86,86,86,86,86,86,86,86,86,86,86,86,86,86,86,86,86,86,86,86,86,86,86,86,86,86,86,86,86,86,86,86,86,86,86,86,86,86,86,86,86,86,86,86,86,86,86,86,86,86,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,78,78,78,78,78,78,78,78,78,78,78,78,78,78,78,78,78,78,78,78,78,78,78,78,78,78,78,78,78,78,78,78,78,78,78,78,78,78,78,78,78,78,78,78,78,78,78,78,78,78,78,78,78,78,78,78,78,78,78,78,78,78,78,78,78,78,78,78,78,78,78,78,78,78,78,78,78,78,78,78,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,98,98,98,98,98,98,98,98,98,98,98,98,98,98,98,98,98,98,98,98,98,98,98,98,98,98,98,98,98,98,98,98,98,98,98,98,98,98,98,98,98,98,98,98,98,98,98,98,98,98,98,98,98,98,98,98,98,98,98,98,98,98,98,98,98,98,98,98,98,98,98,98,98,98,98,98,98,98,98,98,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,59,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,73,73,73,73,73,73,73,73,73,73,73,73,73,73,73,73,73,73,73,73,73,73,73,73,73,73,73,73,73,73,73,73,73,73,73,73,73,73,73,73,73,73,73,73,73,73,73,73,73,73,73,73,73,73,73,73,73,73,73,73,73,73,73,73,73,73,73,73,73,73,73,73,73,73,73,73,73,73,73,73,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,92,92,92,92,92,92,92,92,92,92,92,92,92,92,92,92,92,92,92,92,92,92,92,92,92,92,92,92,92,92,92,92,92,92,92,92,92,92,92,92,92,92,92,92,92,92,92,92,92,92,92,92,92,92,92,92,92,92,92,92,92,92,92,92,92,92,92,92,92,92,92,92,92,92,92,92,92,92,92,92,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,77,77,77,77,77,77,77,77,77,77,77,77,77,77,77,77,77,77,77,77,77,77,77,77,77,77,77,77,77,77,77,77,77,77,77,77,77,77,77,77,77,77,77,77,77,77,77,77,77,77,77,77,77,77,77,77,77,77,77,77,77,77,77,77,77,77,77,77,77,77,77,77,77,77,77,77,77,77,77,77,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,49,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,76,76,76,76,76,76,76,76,76,76,76,76,76,76,76,76,76,76,76,76,76,76,76,76,76,76,76,76,76,76,76,76,76,76,76,76,76,76,76,76,76,76,76,76,76,76,76,76,76,76,76,76,76,76,76,76,76,76,76,76,76,76,76,76,76,76,76,76,76,76,76,76,76,76,76,76,76,76,76,76,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,84,84,84,84,84,84,84,84,84,84,84,84,84,84,84,84,84,84,84,84,84,84,84,84,84,84,84,84,84,84,84,84,84,84,84,84,84,84,84,84,84,84,84,84,84,84,84,84,84,84,84,84,84,84,84,84,84,84,84,84,84,84,84,84,84,84,84,84,84,84,84,84,84,84,84,84,84,84,84,84,97,97,97,97,97,97,97,97,97,97,97,97,97,97,97,97,97,97,97,97,97,97,97,97,97,97,97,97,97,97,97,97,97,97,97,97,97,97,97,97,97,97,97,97,97,97,97,97,97,97,97,97,97,97,97,97,97,97,97,97,97,97,97,97,97,97,97,97,97,97,97,97,97,97,97,97,97,97,97,97,75,75,75,75,75,75,75,75,75,75,75,75,75,75,75,75,75,75,75,75,75,75,75,75,75,75,75,75,75,75,75,75,75,75,75,75,75,75,75,75,75,75,75,75,75,75,75,75,75,75,75,75,75,75,75,75,75,75,75,75,75,75,75,75,75,75,75,75,75,75,75,75,75,75,75,75,75,75,75,75,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,44,44,44,44,44,44,44,44,44,44,44,44,44,44,44,44,44,44,44,44,44,44,44,44,44,44,44,44,44,44,44,44,44,44,44,44,44,44,44,44,44,44,44,44,44,44,44,44,44,44,44,44,44,44,44,44,44,44,44,44,44,44,44,44,44,44,44,44,44,44,44,44,44,44,44,44,44,44,44,44,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,93,93,93,93,93,93,93,93,93,93,93,93,93,93,93,93,93,93,93,93,93,93,93,93,93,93,93,93,93,93,93,93,93,93,93,93,93,93,93,93,93,93,93,93,93,93,93,93,93,93,93,93,93,93,93,93,93,93,93,93,93,93,93,93,93,93,93,93,93,93,93,93,93,93,93,93,93,93,93,93,36,36,36,36,36,36,36,36,36,36,36,36,36,36,36,36,36,36,36,36,36,36,36,36,36,36,36,36,36,36,36,36,36,36,36,36,36,36,36,36,36,36,36,36,36,36,36,36,36,36,36,36,36,36,36,36,36,36,36,36,36,36,36,36,36,36,36,36,36,36,36,36,36,36,36,36,36,36,36,36,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,71,71,71,71,71,71,71,71,71,71,71,71,71,71,71,71,71,71,71,71,71,71,71,71,71,71,71,71,71,71,71,71,71,71,71,71,71,71,71,71,71,71,71,71,71,71,71,71,71,71,71,71,71,71,71,71,71,71,71,71,71,71,71,71,71,71,71,71,71,71,71,71,71,71,71,71,71,71,71,71,83,83,83,83,83,83,83,83,83,83,83,83,83,83,83,83,83,83,83,83,83,83,83,83,83,83,83,83,83,83,83,83,83,83,83,83,83,83,83,83,83,83,83,83,83,83,83,83,83,83,83,83,83,83,83,83,83,83,83,83,83,83,83,83,83,83,83,83,83,83,83,83,83,83,83,83,83,83,83,83,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,96,96,96,96,96,96,96,96,96,96,96,96,96,96,96,96,96,96,96,96,96,96,96,96,96,96,96,96,96,96,96,96,96,96,96,96,96,96,96,96,96,96,96,96,96,96,96,96,96,96,96,96,96,96,96,96,96,96,96,96,96,96,96,96,96,96,96,96,96,96,96,96,96,96,96,96,96,96,96,96,65,65,65,65,65,65,65,65,65,65,65,65,65,65,65,65,65,65,65,65,65,65,65,65,65,65,65,65,65,65,65,65,65,65,65,65,65,65,65,65,65,65,65,65,65,65,65,65,65,65,65,65,65,65,65,65,65,65,65,65,65,65,65,65,65,65,65,65,65,65,65,65,65,65,65,65,65,65,65,65,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,79,79,79,79,79,79,79,79,79,79,79,79,79,79,79,79,79,79,79,79,79,79,79,79,79,79,79,79,79,79,79,79,79,79,79,79,79,79,79,79,79,79,79,79,79,79,79,79,79,79,79,79,79,79,79,79,79,79,79,79,79,79,79,79,79,79,79,79,79,79,79,79,79,79,79,79,79,79,79,79,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,45,45,45,45,45,45,45,45,45,45,45,45,45,45,45,45,45,45,45,45,45,45,45,45,45,45,45,45,45,45,45,45,45,45,45,45,45,45,45,45,45,45,45,45,45,45,45,45,45,45,45,45,45,45,45,45,45,45,45,45,45,45,45,45,45,45,45,45,45,45,45,45,45,45,45,45,45,45,45,45,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,55,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,57,88,88,88,88,88,88,88,88,88,88,88,88,88,88,88,88,88,88,88,88,88,88,88,88,88,88,88,88,88,88,88,88,88,88,88,88,88,88,88,88,88,88,88,88,88,88,88,88,88,88,88,88,88,88,88,88,88,88,88,88,88,88,88,88,88,88,88,88,88,88,88,88,88,88,88,88,88,88,88,88,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,26,26,26,26,26,26,26,26,26,26,26,26,26,26,26,26,26,26,26,26,26,26,26,26,26,26,26,26,26,26,26,26,26,26,26,26,26,26,26,26,26,26,26,26,26,26,26,26,26,26,26,26,26,26,26,26,26,26,26,26,26,26,26,26,26,26,26,26,26,26,26,26,26,26,26,26,26,26,26,26,68,68,68,68,68,68,68,68,68,68,68,68,68,68,68,68,68,68,68,68,68,68,68,68,68,68,68,68,68,68,68,68,68,68,68,68,68,68,68,68,68,68,68,68,68,68,68,68,68,68,68,68,68,68,68,68,68,68,68,68,68,68,68,68,68,68,68,68,68,68,68,68,68,68,68,68,68,68,68,68,67,67,67,67,67,67,67,67,67,67,67,67,67,67,67,67,67,67,67,67,67,67,67,67,67,67,67,67,67,67,67,67,67,67,67,67,67,67,67,67,67,67,67,67,67,67,67,67,67,67,67,67,67,67,67,67,67,67,67,67,67,67,67,67,67,67,67,67,67,67,67,67,67,67,67,67,67,67,67,67,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,61,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,91,91,91,91,91,91,91,91,91,91,91,91,91,91,91,91,91,91,91,91,91,91,91,91,91,91,91,91,91,91,91,91,91,91,91,91,91,91,91,91,91,91,91,91,91,91,91,91,91,91,91,91,91,91,91,91,91,91,91,91,91,91,91,91,91,91,91,91,91,91,91,91,91,91,91,91,91,91,91,91,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,95,95,95,95,95,95,95,95,95,95,95,95,95,95,95,95,95,95,95,95,95,95,95,95,95,95,95,95,95,95,95,95,95,95,95,95,95,95,95,95,95,95,95,95,95,95,95,95,95,95,95,95,95,95,95,95,95,95,95,95,95,95,95,95,95,95,95,95,95,95,95,95,95,95,95,95,95,95,95,95,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,72,72,72,72,72,72,72,72,72,72,72,72,72,72,72,72,72,72,72,72,72,72,72,72,72,72,72,72,72,72,72,72,72,72,72,72,72,72,72,72,72,72,72,72,72,72,72,72,72,72,72,72,72,72,72,72,72,72,72,72,72,72,72,72,72,72,72,72,72,72,72,72,72,72,72,72,72,72,72,72,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,81,81,81,81,81,81,81,81,81,81,81,81,81,81,81,81,81,81,81,81,81,81,81,81,81,81,81,81,81,81,81,81,81,81,81,81,81,81,81,81,81,81,81,81,81,81,81,81,81,81,81,81,81,81,81,81,81,81,81,81,81,81,81,81,81,81,81,81,81,81,81,81,81,81,81,81,81,81,81,81,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,74,74,74,74,74,74,74,74,74,74,74,74,74,74,74,74,74,74,74,74,74,74,74,74,74,74,74,74,74,74,74,74,74,74,74,74,74,74,74,74,74,74,74,74,74,74,74,74,74,74,74,74,74,74,74,74,74,74,74,74,74,74,74,74,74,74,74,74,74,74,74,74,74,74,74,74,74,74,74,74,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,66,66,66,66,66,66,66,66,66,66,66,66,66,66,66,66,66,66,66,66,66,66,66,66,66,66,66,66,66,66,66,66,66,66,66,66,66,66,66,66,66,66,66,66,66,66,66,66,66,66,66,66,66,66,66,66,66,66,66,66,66,66,66,66,66,66,66,66,66,66,66,66,66,66,66,66,66,66,66,66,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,47,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6]
,[[2,7922],[194,7885],[278,7899],[145,7855],[281,7402],[467,7538],[209,7788],[117,7645],[785,7786],[144,7773],[30,7820],[114,7625],[726,7605],[226,7326],[432,7636],[626,7594],[571,7682],[744,7892],[312,7209],[254,7411],[1,7460],[382,7485],[258,7265],[19,7598],[101,7646],[400,7470],[594,7675],[4,7231],[702,7759],[294,7497],[752,7539],[565,7228],[102,7537],[746,7871],[97,7701],[416,7506],[146,7794],[422,7909],[174,7728],[3,7800],[132,7502],[98,7321],[367,7206],[325,7218],[669,7824],[417,7830],[586,7922],[425,7301],[748,7936],[791,7399],[125,7537],[166,7925],[339,7581],[163,7914],[250,7913],[166,7336],[676,7765],[49,7532],[515,7761],[197,7666],[663,7655],[651,7511],[240,7558],[277,7957],[58,7916],[730,7513],[508,7614],[178,7548],[764,7892],[203,7810],[372,7247],[790,7309],[89,7318],[337,7994],[162,7532],[327,7718],[702,7645],[185,7674],[456,7921],[556,7765],[551,7381],[411,7241],[226,7284],[455,7482],[544,7801],[19,7443],[20,7460],[367,7746],[519,7207],[723,7977],[607,7682],[120,7756],[442,7269],[316,7334],[182,7823],[225,7576],[719,7536],[708,7986],[302,7876],[748,7573],[610,7385],[46,7533],[754,7561],[577,7616],[453,7528],[455,7707],[613,7821],[90,7309],[696,7392],[459,7806],[561,7428],[18,7925],[312,7976],[339,7678],[421,7346],[452,7456],[332,7985],[699,7306],[89,7598],[0,7762],[583,7565],[543,7784],[475,7239],[774,7311],[356,7875],[171,7385],[119,7995],[210,7406],[388,7926],[534,7809],[795,7584],[62,7207],[526,7400],[278,7669],[443,7897],[766,7272],[133,7887],[519,7850],[591,7728],[600,7832],[738,7432],[7,7903],[507,7487],[399,7308],[758,7319],[765,7452],[748,7785],[618,7973],[654,7690],[591,7808],[129,7677],[107,7848],[336,7912],[403,7249],[225,7416],[533,7893],[497,7966],[553,7731],[692,7561],[149,7725],[374,7951],[658,7774],[629,7724],[406,7724],[667,7307],[559,7991],[785,7464],[794,7440],[589,7678],[405,7913],[248,7252],[728,7782],[8,7860],[4,7681],[242,7708],[601,7491],[630,7796],[626,7945],[669,7504],[800,7483],[320,7673],[700,7857],[675,7458],[317,7483],[112,7244],[462,7612],[595,7374],[761,7636],[138,7594],[655,7989],[351,7491],[669,7437],[304,7987],[462,7563],[581,7311],[705,7600],[31,7951],[265,7455],[349,7793],[667,7727],[214,7825],[528,7537],[641,7852],[727,7781],[758,7255],[473,7820],[720,7786],[439,7572],[518,7521],[461,7719],[774,7428],[688,7708],[337,7626],[681,7765],[73,7587],[488,7738],[316,7423],[313,7893],[39,7700],[710,7415],[58,7653],[667,7886],[100,7903],[477,7676],[397,7296],[702,7852],[197,7523],[673,7226],[657,7841],[2,7703],[520,7747],[223,7597],[597,7360],[152,7395],[202,7932],[728,7713],[329,7489],[13,7441],[193,7618],[595,7801],[618,7548],[133,7767],[341,7898],[784,7244],[745,7391],[229,7289],[783,7341],[699,7602],[322,7400],[162,7669],[243,7521],[711,7887],[727,7538],[121,7491],[638,7661],[159,7468],[608,7906],[54,7392],[143,7331],[251,7474],[493,7698],[76,7499],[434,7827],[337,7234],[599,7448],[727,7565],[321,7576],[352,7911],[462,7242],[152,7237],[41,7620],[116,7681],[257,7275],[199,7628],[316,7471],[369,7777],[280,7681],[401,7313],[784,7363],[110,7764],[577,7455],[678,7811],[95,7850],[151,7689],[693,7949],[773,7214],[428,7532],[30,7648],[752,7614],[105,7357],[172,7353],[554,7864],[242,7871],[727,7847],[27,7573],[395,7503],[290,7629],[570,7911],[361,7454],[687,7204],[262,7579],[378,7531],[348,7794],[340,7252],[129,7617],[217,7922],[340,7638],[6,7711],[510,7698],[786,7751],[620,7888],[319,7421],[237,7861],[140,7699],[419,7396],[590,7940],[615,7263],[269,7894],[420,7236],[29,7732],[182,7910],[759,7450],[409,7779],[386,7938],[676,7345],[434,7776],[17,7388],[702,7388],[153,7263],[577,7654],[383,7545],[692,7223],[410,7306],[468,7991],[620,7238],[442,7858],[473,7224],[738,7268],[348,7894],[110,7376],[39,7257],[212,7798],[213,7458],[316,7826],[33,7668],[446,7813],[487,7367],[117,7591],[186,7945],[75,7764],[664,7398],[702,7443],[773,7902],[236,7802],[752,7850],[74,7958],[610,7361],[591,7586],[664,7824],[150,7315],[2,7545],[229,7225],[491,7287],[163,7377],[427,7243],[56,7772],[128,7348],[523,7438],[668,7674],[455,7966],[286,7386],[464,7998],[55,7523],[292,7835],[573,7531],[502,7952],[554,7748],[455,7370],[560,7348],[63,7609],[198,7354],[162,7909],[676,7730],[710,7965],[760,7226],[81,7371],[431,7790],[566,7254],[298,7464],[425,7920],[429,7574],[515,7793],[331,7340],[758,7477],[502,7618],[13,7781],[426,7909],[19,7862],[436,7471],[450,7359],[711,7842],[263,7550],[181,7570],[472,7422],[463,7763],[476,7911],[256,7680],[506,7760],[549,7467],[14,7652],[3,7913],[715,7820],[279,7626],[325,7690],[479,7669],[794,7254],[444,7288],[145,7829],[526,7712],[621,7664],[156,7758],[313,7358],[408,7734],[62,7813],[139,7724],[150,7610],[703,7936],[72,7924],[659,7974],[386,7463],[122,7648],[27,7201],[349,7400],[504,7276],[660,7728],[434,7234],[342,7596],[746,7934],[407,7755],[527,7210],[206,7276],[706,7856],[411,7587],[312,7683],[591,7576],[651,7265],[147,7677],[596,7736],[195,7326],[178,7979],[780,7468],[136,7770],[701,7569],[243,7582],[688,7902],[276,7285],[408,7218],[86,7790],[618,7287],[631,7769],[697,7467],[569,7220],[534,7650],[2,7919],[121,7682],[1,7908],[526,7998],[741,7901],[409,7697],[105,7575],[174,7240],[491,7678],[664,7305],[741,7256],[482,7492],[59,7491],[231,7430],[434,7761],[592,7581],[749,7653],[611,7588],[463,7916],[155,7863],[300,7622],[85,7885],[54,7535],[68,7293],[35,7784],[0,7252],[692,7680],[52,7308],[508,7768],[754,7570],[28,7943],[180,7212],[381,7249],[152,7556],[681,7495],[578,7684],[167,7902],[780,7841],[235,7920],[699,7832],[179,7233],[83,7212],[690,7376],[634,7681],[373,7839],[617,7684],[275,7863],[471,7890],[709,7869],[679,7541],[655,7435],[613,7632],[329,7809],[766,7662],[766,7516],[260,7978],[171,7554],[154,7592],[679,7894],[696,7830],[781,7959],[264,7885],[664,7369],[57,7488],[85,7614],[291,7315],[357,7997],[736,7416],[451,7498],[40,7896],[375,7295],[192,7444],[469,7999],[627,7985],[235,7654],[433,7969],[404,7959],[587,7989],[703,7822],[458,7588],[546,7317],[761,7892],[385,7985],[4,7742],[327,7551],[508,7985],[90,7668],[308,7813],[673,7247],[371,7733],[305,7830],[740,7600],[528,7721],[80,7481],[279,7802],[626,7235],[527,7259],[216,7545],[515,7610],[175,7490],[752,7246],[690,7564],[454,7461],[271,7538],[27,7778],[632,7791],[571,7647],[214,7387],[264,7582],[783,7449],[324,7348],[451,7954],[605,7311],[99,7319],[92,7908],[58,7917],[530,7580],[709,7272],[547,7456],[420,7893],[434,7723],[624,7767],[684,7944],[329,7810],[227,7369],[78,7448],[365,7790],[421,7320],[512,7881],[214,7253],[674,7664],[740,7911],[203,7866],[281,7909],[544,7208],[777,7240],[159,7351],[672,7628],[271,7358],[354,7830],[507,7878],[137,7208],[342,7539],[303,7759],[167,7788],[529,7695],[561,7304],[735,7824],[144,7265],[776,7488],[481,7899],[261,7823],[746,7752],[286,7657],[312,7595],[5,7501],[178,7797],[253,7666],[106,7693],[738,7760],[258,7898],[485,7384],[82,7468],[440,7280],[469,7387],[511,7544],[548,7248],[277,7552],[147,7836],[185,7633],[456,7638],[634,7715],[530,7864],[633,7665],[603,7754],[537,7672],[481,7538],[300,7485],[337,7725],[264,7662],[776,7501],[70,7441],[284,7204],[118,7825],[139,7939],[595,7908],[218,7625],[3,7907],[336,7816],[445,7386],[712,7664],[340,7879],[65,7915],[595,7413],[257,7585],[255,7728],[617,7227],[113,7356],[285,7740],[293,7501],[490,7818],[248,7632],[736,7885],[357,7616],[121,7271],[368,7619],[567,7611],[139,7922],[135,7918],[245,7263],[476,7228],[673,7628],[343,7828],[258,7730],[30,7523],[237,7304],[387,7235],[134,7923],[133,7660],[650,7506],[247,7454],[54,7418],[450,7891],[180,7866],[732,7968],[503,7269],[549,7623],[650,7864],[165,7491],[305,7667],[360,7628],[192,7264],[107,7333],[640,7278],[214,7756],[715,7214],[503,7757],[190,7770],[499,7752],[776,7248],[404,7626],[79,7526],[713,7505],[355,7837],[522,7539],[525,7610],[385,7437],[362,7643],[333,7925],[303,7281],[535,7229],[169,7843],[564,7896],[534,7380],[346,7342],[77,7375],[678,7735],[721,7753],[390,7643],[598,7242],[323,7828],[279,7599],[603,7931],[620,7738],[779,7517],[179,7749],[513,7360],[709,7745],[293,7659],[36,7536],[157,7245],[248,7281],[685,7425],[690,7371],[631,7753],[568,7671],[93,7951],[772,7371],[214,7963],[163,7217],[212,7642],[390,7735],[446,7500],[129,7653],[766,7692],[558,7598],[139,7637],[554,7919],[679,7996],[346,7390],[482,7347],[541,7673],[756,7587],[29,7939],[249,7353],[6,7717],[494,7659],[574,7431],[275,7648],[81,7961],[7,7616],[186,7881],[549,7433],[227,7472],[653,7848],[355,7972],[774,7851],[721,7577],[491,7448],[350,7635],[177,7397],[665,7332],[463,7711],[659,7930],[785,7237],[57,7460],[338,7418],[347,7743],[365,7922],[598,7681],[223,7262],[128,7994],[473,7425],[763,7851],[685,7329],[174,7763],[352,7658],[23,7519],[726,7367],[260,7242],[781,7944],[156,7539],[350,7545],[297,7620],[533,7818],[743,7917],[700,7428],[592,7660],[469,7864],[39,7500],[106,7306],[425,7688],[571,7750],[70,7568],[597,7631],[652,7492],[409,7860],[471,7438],[540,7600],[351,7589],[623,7210],[304,7485],[527,7887],[270,7964],[687,7641],[511,7964],[233,7412],[191,7928],[182,7838],[72,7906],[166,7490],[557,7816],[338,7560],[547,7676],[706,7648],[29,7210],[675,7633],[165,7444],[693,7829],[268,7520],[596,7807],[693,7485],[163,7631],[281,7232],[499,7622],[24,7851],[568,7808],[330,7411],[60,7287],[131,7935],[48,7430],[250,7789],[523,7229],[710,7320],[247,7938],[516,7710],[93,7977],[384,7347],[557,7429],[253,7564],[434,7762],[639,7254],[328,7761],[793,7277],[39,7753],[689,7217],[119,7574],[223,7411],[468,7486],[202,7533],[413,7669],[21,7332],[651,7871],[460,7702],[169,7723],[785,7671],[72,7526],[390,7574],[622,7641],[675,7835],[376,7673],[713,7795],[683,7502],[361,7706],[348,7781],[351,7826],[583,7579],[376,7226],[350,7531],[664,7445],[704,7844],[396,7337],[576,7927],[646,7284],[453,7317],[307,7726],[373,7493],[76,7772],[229,7839],[353,7364],[706,7520],[134,7956],[678,7575],[146,7263],[464,7884],[792,7742],[660,7712],[167,7993],[777,7276],[403,7865],[320,7298],[68,7824],[644,7926],[189,7294],[306,7583],[503,7502],[727,7502],[327,7352],[61,7409],[41,7880],[138,7684],[645,7594],[783,7837],[268,7674],[207,7226],[525,7948],[793,7760],[144,7507],[58,7393],[746,7571],[699,7447],[176,7940],[93,7372],[755,7491],[292,7209],[754,7919],[177,7796],[205,7991],[446,7439],[296,7859],[22,7789],[256,7292],[180,7933],[90,7764],[39,7570],[735,7875],[168,7562],[489,7785],[324,7796],[581,7262],[724,7227],[289,7689],[622,7458],[90,7339],[671,7442],[157,7532],[309,7616],[8,7288],[566,7808],[89,7316],[208,7509],[758,7947],[238,7468],[514,7927],[375,7684],[70,7406],[219,7815],[785,7902],[638,7642],[708,7380],[472,7637],[559,7625],[89,7390],[505,7444],[297,7331],[118,7986],[119,7354],[26,7991],[169,7834],[88,7912],[724,7408],[514,7285],[229,7240],[94,7496],[606,7512],[573,7235],[166,7832],[538,7816],[51,7769],[558,7860],[540,7928],[71,7792],[388,7893],[422,7509],[369,7361],[192,7319],[586,7310],[692,7789],[604,7236],[525,7357],[71,7886],[222,7396],[328,7601],[575,7944],[712,7411],[96,7652],[155,7439],[778,7641],[641,7608],[500,7255],[307,7928],[712,7270],[495,7287],[11,7870],[222,7907],[561,7776],[94,7833],[117,7718],[781,7690],[297,7236],[328,7230],[114,7444],[739,7410],[800,7362],[23,7981],[254,7478],[269,7806],[376,7640],[28,7704],[78,7462],[531,7686],[788,7600],[353,7259],[732,7707],[464,7555],[769,7920],[157,7566],[724,7868],[470,7355],[231,7277],[386,7974],[764,7413],[120,7603],[26,7400],[678,7709],[657,7715],[745,7278],[95,7867],[687,7962],[787,7256],[323,7989],[706,7973],[450,7231],[192,7977],[587,7826],[388,7990],[210,7261],[434,7608],[472,7223],[13,7553],[608,7822],[287,7612],[611,7606],[406,7902],[619,7451],[732,7340],[784,7453],[223,7906],[472,7260],[29,7704],[110,7960],[697,7525],[535,7209],[800,7227],[593,7707],[67,7480],[671,7249],[572,7630],[202,7455],[186,7421],[305,7420],[31,7970],[415,7545],[547,7996],[502,7500],[292,7912],[554,7910],[538,7765],[197,7838],[666,7375],[249,7679],[343,7814],[749,7205],[591,7704],[190,7345],[537,7532],[554,7909],[144,7902],[382,7763],[345,7960],[280,7564],[323,7315],[216,7835],[787,7317],[589,7630],[800,7224],[219,7518],[531,7369],[796,7969],[240,7638],[507,7755],[574,7411],[560,7927],[500,7400],[405,7776],[650,7397],[148,7446],[544,7217],[33,7464],[711,7441],[421,7461],[449,7572],[531,7518],[423,7631],[280,7729],[441,7955],[491,7928],[272,7300],[402,7436],[502,7985],[90,7333],[39,7609],[233,7308],[468,7540],[353,7372],[588,7652],[492,7609],[38,7324],[266,7566],[525,7509],[4,7725],[105,7313],[767,7994],[275,7459],[81,7578],[449,7382],[557,7315],[723,7519],[728,7419],[719,7588],[70,7274],[424,7751],[390,7321],[104,7514],[395,7231],[203,7584],[77,7291],[528,7822],[170,7609],[766,7966],[22,7540],[349,7547],[726,7845],[276,7588],[359,7464],[179,7959],[407,7712],[252,7588],[518,7971],[261,7997],[237,7711],[462,7291],[273,7227],[203,7497],[692,7479],[647,7902],[322,7487],[346,7780],[362,7890],[420,7634],[29,7209],[236,7207],[88,7294],[615,7623],[658,7340],[155,7708],[794,7353],[225,7735],[385,7801],[482,7473],[703,7328],[707,7301],[667,7944],[784,7655],[761,7412],[399,7743],[237,7747],[376,7873],[56,7794],[332,7895],[646,7661],[21,7799],[198,7464],[30,7999],[733,7819],[587,7287],[416,7899],[369,7233],[556,7316],[71,7393],[421,7909],[265,7971],[792,7219],[246,7481],[540,7843],[257,7775],[278,7393],[763,7685],[474,7234],[15,7907],[430,7376],[498,7784],[338,7929],[676,7790],[343,7926],[127,7512],[185,7808],[363,7461],[291,7827],[766,7596],[438,7789],[337,7500],[141,7286],[759,7969],[713,7687],[712,7914],[362,7645],[133,7686],[351,7372],[512,7207],[12,7314],[424,7960],[735,7268],[663,7657],[209,7477],[313,7672],[593,7878],[291,7913],[769,7526],[599,7388],[239,7520],[690,7320],[419,7928],[217,7764],[233,7366],[432,7485],[749,7726],[620,7698],[735,7293],[703,7899],[47,7254],[92,7655],[13,7676],[569,7305],[128,7899],[729,7837],[290,7851],[671,7545],[459,7614],[76,7955],[756,7445],[41,7477],[796,7779],[308,7772],[288,7292],[715,7495],[177,7840],[722,7852],[251,7520],[730,7559],[792,7353],[554,7406],[655,7418],[494,7373],[690,7747],[483,7930],[367,7615],[674,7703],[587,7768],[127,7875],[213,7249],[141,7624],[333,7587],[280,7852],[467,7928],[483,7298],[84,7549],[530,7773],[133,7946],[746,7266],[537,7679],[38,7891],[534,7808],[284,7274],[738,7202],[499,7945],[590,7323],[118,7626],[122,7277],[307,7610],[263,7460],[390,7929],[665,7283],[105,7796],[516,7446],[61,7976],[590,7668],[371,7510],[176,7949],[785,7509],[501,7417],[749,7988],[686,7728],[559,7867],[783,7544],[727,7389],[246,7708],[738,7885],[494,7499],[572,7890],[745,7707],[469,7204],[57,7680],[573,7773],[123,7847],[150,7548],[248,7376],[342,7297],[160,7288],[768,7865],[744,7989],[568,7231],[563,7367],[773,7369],[415,7694],[585,7799],[61,7829],[418,7881],[415,7621],[745,7219],[151,7754],[440,7569],[736,7408],[562,7420],[584,7862],[369,7670],[64,7619],[314,7612],[776,7884],[542,7336],[111,7986],[734,7201],[14,7393],[699,7962],[483,7209],[775,7967],[607,7221],[457,7247],[251,7699],[532,7995],[62,7740],[646,7724],[769,7950],[426,7930],[721,7982],[445,7969],[593,7430],[168,7237],[510,7970],[723,7235],[584,7727],[240,7479],[516,7477],[608,7358],[688,7911],[708,7645],[503,7939],[311,7753],[450,7619],[700,7380],[630,7619],[355,7260],[486,7493],[689,7475],[348,7604],[324,7652],[136,7637],[482,7604],[369,7893],[282,7291],[791,7812],[522,7364],[514,7305],[457,7800],[277,7710],[488,7791],[358,7203],[41,7398],[314,7541],[783,7335],[772,7352],[62,7390],[447,7383],[406,7826],[231,7259],[221,7328],[761,7598],[679,7461],[353,7375],[380,7653],[228,7447],[10,7826],[732,7977],[33,7672],[771,7993],[264,7994],[778,7240],[750,7590],[232,7543],[88,7643],[527,7551],[150,7862],[152,7478],[476,7609],[762,7690],[376,7325],[128,7550],[725,7771],[204,7693],[778,7848],[588,7807],[25,7263],[737,7566],[214,7609],[740,7985],[417,7814],[213,7398],[139,7704],[444,7436],[222,7464],[132,7353],[545,7563],[713,7609],[163,7991],[789,7732],[236,7676],[143,7424],[509,7890],[180,7390],[345,7968],[593,7412],[104,7236],[238,7434],[734,7682],[785,7950],[715,7931],[54,7207],[556,7909],[598,7519],[145,7870],[212,7457],[34,7409],[760,7656],[11,7473],[202,7395],[238,7979],[201,7899],[126,7847],[641,7750],[526,7415],[546,7487],[375,7608],[649,7984],[84,7330],[368,7861],[498,7760],[640,7212],[275,7675],[224,7650],[103,7806],[536,7927],[657,7841],[380,7970],[703,7893],[116,7688],[431,7785],[674,7594],[553,7216],[668,7842],[395,7502],[378,7922],[365,7514],[309,7945],[707,7347],[98,7576],[501,7881],[713,7752],[227,7674],[635,7884],[161,7645],[296,7230],[330,7411],[730,7348],[644,7661],[202,7863],[616,7638],[534,7742],[371,7831],[209,7469],[743,7335],[538,7769],[218,7850],[585,7238],[226,7817],[467,7828],[621,7679],[693,7888],[530,7422],[693,7416],[82,7910],[381,7223],[243,7368],[96,7808],[522,7583],[775,7795],[690,7630],[353,7571],[798,7282],[697,7901],[744,7230],[130,7539],[11,7605],[446,7644],[754,7864],[536,7614],[653,7887],[785,7237],[412,7861],[468,7428],[304,7706],[427,7682],[95,7504],[581,7790],[694,7744],[438,7452],[145,7772],[597,7566],[741,7830],[91,7948],[0,7436],[540,7542],[189,7517],[150,7680],[390,7372],[156,7305],[333,7560],[579,7977],[410,7447],[693,7652],[302,7410],[586,7892],[424,7433],[610,7818],[263,7829],[344,7810],[295,7207],[205,7769],[576,7556],[328,7961],[292,7624],[370,7645],[797,7878],[668,7576],[203,7276],[481,7259],[525,7702],[222,7232],[54,7353],[756,7293],[16,7878],[149,7953],[478,7408],[389,7887],[385,7426],[699,7960],[171,7559],[753,7599],[672,7672],[209,7937],[721,7390],[174,7947],[209,7491],[517,7662],[274,7777],[218,7812],[66,7636],[683,7600],[5,7298],[594,7326],[34,7817],[10,7509],[561,7731],[441,7605],[737,7944],[23,7965],[452,7412],[763,7357],[423,7516],[311,7973],[458,7992],[787,7771],[584,7843],[707,7402],[680,7680],[526,7565],[628,7571],[645,7486],[93,7695],[441,7259],[631,7886],[23,7813],[202,7707],[99,7624],[299,7989],[724,7872],[677,7574],[653,7772],[110,7214],[292,7558],[769,7702],[252,7932],[278,7628],[445,7788],[307,7461],[505,7418],[705,7439],[552,7651],[786,7762],[58,7962],[789,7928],[333,7333],[513,7786],[30,7875],[339,7598],[447,7553],[89,7766],[195,7653],[555,7510],[757,7660],[453,7292],[607,7741],[771,7752],[514,7442],[455,7480],[251,7470],[363,7244],[620,7612],[520,7526],[633,7419],[515,7241],[404,7361],[764,7531],[59,7353],[245,7246],[523,7784],[660,7805],[689,7599],[287,7602],[622,7855],[169,7471],[227,7505],[793,7437],[366,7457],[541,7615],[609,7537],[181,7314],[191,7304],[702,7564],[398,7232],[240,7716],[462,7265],[5,7733],[559,7246],[214,7538],[317,7629],[636,7497],[390,7637],[0,7506],[601,7260],[217,7688],[412,7465],[451,7482],[470,7640],[752,7796],[125,7872],[759,7283],[436,7283],[190,7381],[118,7546],[45,7972],[243,7944],[588,7896],[414,7915],[320,7559],[564,7618],[727,7714],[728,7918],[71,7853],[576,7854],[45,7243],[309,7325],[321,7733],[616,7426],[220,7421],[381,7334],[556,7625],[731,7932],[677,7610],[1,7302],[228,7727],[321,7523],[767,7200],[784,7226],[171,7800],[346,7693],[197,7731],[11,7308],[31,7270],[188,7501],[362,7676],[96,7754],[94,7583],[572,7256],[394,7965],[669,7292],[612,7502],[622,7781],[789,7792],[118,7394],[410,7840],[741,7582],[37,7564],[344,7323],[589,7297],[687,7957],[607,7427],[590,7621],[545,7648],[83,7713],[443,7876],[504,7278],[494,7207],[549,7211],[657,7518],[624,7351],[611,7408],[724,7366],[769,7771],[732,7574],[17,7702],[718,7303],[614,7632],[85,7357],[605,7425],[212,7343],[76,7980],[199,7486],[590,7561],[167,7338],[43,7776],[790,7471],[42,7881],[53,7418],[489,7892],[74,7682],[507,7571],[534,7769],[357,7594],[158,7933],[459,7461],[553,7207],[397,7609],[108,7269],[707,7385],[424,7958],[134,7938],[33,7865],[235,7260],[397,7806],[379,7275],[439,7525],[502,7889],[95,7372],[112,7320],[567,7641],[49,7858],[64,7755],[83,7324],[683,7947],[134,7884],[727,7860],[798,7435],[695,7829],[623,7426],[200,7260],[186,7532],[365,7797],[763,7720],[787,7707],[594,7937],[138,7260],[439,7354],[226,7317],[291,7878],[349,7814],[440,7768],[673,7904],[517,7917],[619,7307],[20,7796],[720,7596],[648,7310],[584,7981],[414,7760],[49,7495],[367,7231],[352,7367],[91,7266],[144,7972],[50,7422],[432,7206],[531,7325],[414,7480],[794,7305],[705,7770],[787,7213],[259,7461],[765,7616],[43,7754],[718,7277],[708,7825],[511,7304],[262,7239],[793,7765],[241,7479],[435,7777],[333,7469],[287,7779],[360,7230],[413,7211],[604,7896],[757,7540],[513,7235],[423,7334],[708,7667],[268,7548],[407,7642],[608,7770],[225,7900],[633,7852],[559,7556],[585,7869],[41,7698],[264,7345],[622,7480],[80,7512],[146,7881],[633,7873],[16,7497],[635,7567],[465,7866],[453,7641],[183,7523],[195,7890],[779,7842],[546,7494],[465,7615],[770,7599],[619,7287],[614,7553],[199,7245],[98,7600],[292,7707],[142,7737],[645,7610],[126,7680],[300,7888],[729,7584],[531,7527],[596,7214],[137,7902],[396,7332],[623,7656],[397,7286],[123,7976],[399,7596],[140,7987],[736,7753],[408,7645],[8,7986],[194,7515],[611,7734],[130,7476],[409,7761],[608,7846],[691,7622],[494,7866],[47,7327],[405,7578],[387,7361],[509,7963],[502,7900],[302,7245],[161,7264],[372,7709],[135,7411],[88,7934],[151,7530],[151,7480],[183,7407],[348,7355],[349,7960],[399,7234],[0,7733],[328,7963],[535,7834],[316,7814],[399,7264],[463,7384],[110,7570],[57,7547],[791,7353],[8,7840],[368,7949],[233,7538],[203,7985],[89,7392],[736,7321],[527,7478],[380,7623],[17,7537],[627,7838],[235,7495],[679,7352],[248,7353],[129,7943],[735,7928],[126,7206],[619,7287],[150,7792],[277,7464],[762,7431],[742,7311],[283,7915],[656,7276],[287,7231],[551,7697],[524,7427],[271,7911],[201,7897],[650,7676],[493,7805],[780,7757],[154,7905],[311,7246],[774,7352],[98,7317],[437,7743],[150,7433],[94,7422],[503,7837],[579,7897],[763,7729],[82,7714],[111,7868],[15,7243],[697,7780],[392,7354],[318,7366],[418,7552],[474,7731],[790,7831],[772,7645],[222,7895],[340,7570],[516,7684],[462,7489],[794,7555],[130,7627],[218,7985],[719,7507],[514,7735],[545,7347],[200,7966],[775,7705],[641,7352],[322,7518],[302,7965],[190,7331],[654,7529],[207,7534],[119,7269],[477,7739],[695,7770],[775,7391],[566,7225],[339,7247],[255,7457],[344,7574],[235,7767],[753,7739],[191,7723],[483,7362],[7,7468],[451,7606],[484,7645],[388,7587],[725,7245],[545,7331],[553,7759],[507,7506],[328,7923],[713,7712],[457,7280],[740,7881],[132,7256],[640,7683],[320,7412],[73,7900],[220,7933],[399,7768],[368,7818],[194,7305],[380,7995],[768,7926],[139,7605],[131,7297],[672,7870],[35,7692],[302,7928],[84,7463],[269,7425],[3,7850],[124,7215],[396,7283],[284,7493],[364,7345],[605,7811],[391,7513],[360,7924],[472,7519],[277,7409],[218,7270],[609,7827],[490,7299],[125,7899],[371,7432],[232,7448],[283,7794],[461,7368],[637,7642],[630,7678],[29,7294],[25,7968],[46,7964],[210,7848],[309,7431],[301,7508],[614,7814],[508,7573],[756,7928],[555,7917],[364,7764],[734,7623],[641,7270],[751,7743],[536,7718],[94,7687],[524,7613],[54,7927],[133,7318],[375,7786],[604,7357],[511,7821],[261,7422],[467,7647],[224,7727],[130,7702],[295,7797],[46,7415],[219,7320],[164,7371],[190,7434],[616,7529],[270,7770],[41,7649],[14,7496],[461,7411],[702,7320],[319,7949],[755,7737],[223,7605],[568,7674],[317,7670],[482,7940],[508,7461],[6,7670],[18,7426],[412,7594],[420,7980],[361,7611],[12,7203],[688,7660],[386,7934],[142,7365],[562,7358],[123,7385],[703,7860],[551,7330],[123,7794],[217,7715],[425,7750],[168,7692],[568,7467],[780,7650],[136,7636],[267,7925],[220,7813],[210,7402],[11,7506],[739,7997],[368,7859],[600,7641],[583,7370],[678,7978],[37,7611],[410,7875],[406,7233],[279,7756],[360,7437],[96,7386],[207,7458],[669,7470],[210,7548],[533,7944],[283,7792],[575,7293],[115,7733],[673,7516],[443,7559],[100,7357],[267,7385],[121,7229],[398,7967],[240,7678],[570,7898],[90,7685],[775,7924],[19,7514],[572,7927],[743,7430],[385,7458],[640,7741],[97,7285],[705,7620],[744,7546],[593,7558],[370,7955],[565,7840],[726,7581],[586,7464],[170,7275],[637,7452],[511,7787],[772,7402],[106,7413],[781,7619],[198,7905],[403,7544],[590,7708],[181,7700],[409,7672],[437,7732],[504,7687],[499,7574],[581,7414],[369,7772],[435,7273],[213,7994],[316,7291],[497,7463],[609,7274],[598,7554],[225,7993],[568,7224],[560,7868],[557,7504],[183,7642],[503,7333],[196,7405],[622,7821],[13,7895],[501,7615],[117,7572],[28,7233],[96,7276],[83,7429],[624,7308],[342,7515],[538,7200],[315,7489],[187,7967],[203,7622],[257,7711],[633,7360],[776,7276],[267,7799],[618,7213],[89,7202],[743,7737],[403,7299],[393,7995],[256,7671],[720,7820],[18,7519],[643,7874],[617,7784],[91,7550],[492,7429],[543,7877],[27,7590],[465,7827],[586,7590],[472,7882],[236,7596],[462,7385],[37,7798],[691,7603],[121,7246],[185,7979],[198,7669],[185,7642],[312,7429],[690,7377],[58,7572],[521,7596],[581,7779],[449,7362],[273,7571],[635,7680],[106,7880],[770,7861],[372,7570],[698,7654],[632,7400],[160,7986],[500,7678],[7,7321],[118,7897],[343,7515],[694,7833],[98,7677],[692,7711],[210,7840],[187,7947],[423,7204],[600,7634],[520,7548],[416,7720],[565,7904],[548,7856],[703,7763],[430,7972],[64,7235],[774,7736],[351,7889],[619,7685],[205,7555],[512,7587],[490,7805],[300,7417],[713,7399],[420,7480],[278,7708],[52,7372],[391,7405],[153,7701],[405,7704],[673,7634],[783,7559],[608,7603],[270,7953],[342,7730],[455,7871],[759,7596],[394,7833],[265,7797],[648,7471],[453,7962],[80,7557],[362,7843],[755,7444],[281,7653],[258,7527],[663,7236],[734,7853],[132,7459],[24,7993],[373,7447],[453,7707],[173,7814],[662,7205],[572,7942],[279,7895],[765,7736],[625,7607],[322,7432],[657,7838],[264,7540],[27,7442],[451,7557],[286,7767],[33,7991],[493,7737],[7,7583],[235,7744],[221,7403],[746,7746],[380,7612],[594,7393],[191,7938],[621,7323],[253,7616],[615,7588],[310,7958],[537,7243],[713,7483],[799,7229],[486,7426],[469,7414],[56,7481],[89,7560],[1,7939],[395,7423],[180,7923],[263,7230],[321,7305],[46,7955],[144,7617],[369,7728],[235,7745],[462,7846],[595,7642],[210,7406],[298,7604],[478,7487],[663,7637],[272,7908],[403,7642],[763,7238],[253,7423],[139,7697],[201,7784],[634,7685],[270,7314],[343,7808],[150,7254],[725,7976],[589,7490],[71,7254],[90,7221],[183,7843],[500,7646],[45,7694],[546,7312],[698,7810],[335,7317],[401,7506],[530,7468],[606,7672],[487,7779],[649,7680],[207,7730],[201,7441],[22,7637],[89,7469],[257,7420],[329,7657],[671,7580],[276,7922],[484,7740],[604,7415],[349,7223],[352,7884],[775,7986],[656,7902],[187,7619],[32,7390],[778,7678],[29,7943],[413,7740],[481,7542],[443,7974],[481,7261],[492,7287],[651,7916],[554,7896],[122,7363],[614,7386],[244,7936],[408,7297],[223,7783],[112,7330],[192,7703],[472,7459],[372,7812],[660,7802],[533,7504],[688,7304],[716,7954],[212,7454],[510,7621],[512,7656],[9,7542],[710,7717],[797,7211],[501,7792],[67,7976],[45,7843],[771,7614],[656,7572],[121,7230],[437,7473],[41,7707],[36,7665],[540,7409],[779,7750],[697,7860],[308,7651],[543,7948],[275,7724],[200,7210],[233,7460],[529,7987],[359,7673],[218,7643],[336,7875],[291,7906],[694,7207],[298,7468],[635,7332],[490,7389],[526,7259],[629,7210],[193,7320],[462,7767],[518,7450],[679,7599],[262,7451],[757,7284],[438,7704],[762,7731],[333,7964],[639,7714],[273,7467],[273,7242],[564,7775],[674,7786],[323,7960],[601,7420],[24,7404],[190,7840],[34,7672],[641,7580],[446,7205],[730,7550],[158,7633],[634,7839],[799,7255],[127,7961],[370,7362],[211,7864],[38,7333],[621,7812],[674,7837],[411,7590],[200,7424],[21,7814],[169,7451],[366,7609],[361,7471],[3,7450],[30,7650],[611,7858],[607,7320],[361,7481],[61,7384],[699,7674],[213,7367],[26,7320],[149,7843],[623,7565],[220,7295],[571,7534],[4,7222],[299,7331],[205,7280],[680,7744],[143,7916],[39,7976],[709,7593],[529,7329],[56,7508],[541,7327],[800,7909],[33,7780],[243,7257],[354,7504],[543,7494],[725,7659],[222,7728],[367,7654],[774,7697],[614,7657],[209,7800],[538,7966],[153,7477],[733,7537],[653,7896],[335,7442],[538,7307],[383,7286],[548,7360],[497,7348],[694,7392],[783,7727],[141,7647],[281,7334],[600,7563],[100,7716],[351,7389],[371,7573],[662,7678],[702,7628],[792,7740],[538,7877],[741,7281],[344,7546],[605,7242],[249,7254],[525,7326],[588,7633],[745,7944],[635,7536],[310,7468],[759,7713],[464,7596],[105,7891],[346,7215],[725,7980],[301,7724],[567,7488],[599,7957],[117,7930],[333,7954],[31,7722],[585,7339],[670,7461],[295,7616],[728,7497],[42,7443],[340,7227],[454,7622],[609,7569],[354,7560],[615,7914],[792,7415],[377,7540],[520,7723],[79,7263],[519,7974],[728,7337],[679,7356],[457,7649],[794,7537],[307,7657],[786,7662],[642,7708],[504,7478],[381,7788],[184,7578],[290,7925],[120,7329],[222,7726],[458,7351],[295,7954],[600,7454],[494,7810],[481,7830],[39,7217],[505,7263],[727,7491],[338,7572],[614,7491],[560,7517],[582,7904],[754,7626],[472,7471],[736,7308],[446,7636],[135,7402],[694,7777],[349,7631],[674,7700],[120,7357],[337,7572],[600,7426],[351,7230],[172,7995],[461,7641],[457,7469],[639,7951],[21,7584],[308,7326],[158,7994],[156,7702],[151,7895],[494,7606],[369,7402],[64,7489],[376,7315],[710,7842],[206,7482],[105,7708],[652,7250],[57,7547],[289,7573],[344,7769],[246,7496],[361,7564],[637,7544],[372,7645],[191,7325],[434,7771],[679,7419],[208,7581],[673,7695],[716,7982],[424,7624],[521,7255],[573,7827],[364,7976],[7,7508],[761,7794],[93,7230],[514,7627],[777,7934],[52,7500],[205,7281],[485,7290],[379,7537],[696,7682],[572,7528],[784,7362],[95,7840],[735,7792],[467,7797],[237,7428],[400,7769],[392,7247],[797,7830],[698,7448],[70,7884],[696,7630],[36,7736],[589,7730],[518,7759],[358,7861],[695,7225],[90,7234],[377,7580],[780,7765],[201,7811],[80,7295],[110,7578],[209,7377],[44,7365],[415,7491],[49,7233],[625,7925],[194,7902],[694,7951],[243,7924],[222,7784],[443,7448],[511,7801],[43,7389],[2,7881],[685,7441],[556,7773],[651,7658],[531,7373],[670,7264],[573,7745],[629,7416],[540,7504],[290,7660],[532,7301],[770,7974],[422,7991],[463,7642],[413,7454],[765,7479],[138,7956],[8,7756],[238,7506],[524,7916],[94,7390],[301,7390],[760,7984],[414,7645],[299,7692],[284,7352],[745,7622],[314,7481],[30,7463],[474,7863],[572,7398],[784,7900],[783,7370],[779,7706],[613,7644],[416,7445],[452,7808],[666,7535],[530,7295],[699,7909],[562,7685],[454,7509],[279,7824],[580,7372],[318,7597],[539,7656],[679,7290],[313,7463],[393,7824],[222,7782],[631,7920],[342,7455],[94,7534],[338,7647],[413,7518],[669,7347],[200,7829],[285,7210],[156,7841],[387,7797],[777,7211],[568,7778],[384,7474],[784,7803],[397,7599],[428,7883],[531,7265],[124,7877],[550,7386],[142,7790],[800,7349],[248,7880],[790,7865],[645,7861],[139,7472],[392,7773],[502,7340],[190,7253],[231,7678],[212,7435],[497,7923],[571,7992],[276,7254],[396,7891],[729,7265],[208,7727],[423,7703],[308,7923],[533,7534],[507,7713],[250,7252],[271,7537],[335,7798],[394,7902],[333,7891],[794,7660],[678,7608],[667,7882],[380,7671],[514,7594],[575,7893],[2,7255],[361,7545],[68,7952],[134,7605],[652,7966],[490,7867],[434,7467],[654,7461],[616,7470],[219,7624],[229,7369],[478,7237],[376,7620],[109,7461],[444,7563],[197,7457],[259,7381],[43,7262],[298,7354],[89,7901],[522,7641],[453,7456],[466,7932],[102,7577],[140,7251],[255,7533],[84,7846],[588,7294],[585,7740],[355,7886],[673,7222],[348,7618],[383,7842],[103,7441],[229,7524],[560,7527],[50,7756],[89,7840],[269,7392],[190,7390],[726,7776],[512,7687],[433,7620],[727,7409],[190,7701],[687,7805],[708,7933],[48,7998],[285,7331],[212,7409],[561,7298],[1,7611],[41,7961],[517,7830],[79,7954],[502,7244],[39,7572],[154,7329],[617,7283],[541,7473],[357,7670],[670,7997],[302,7794],[785,7616],[370,7832],[612,7221],[341,7335],[484,7219],[720,7384],[359,7831],[635,7639],[97,7315],[296,7420],[324,7715],[198,7245],[319,7206],[642,7695],[490,7949],[759,7936],[375,7326],[226,7241],[438,7278],[11,7809],[769,7356],[143,7311],[411,7762],[601,7666],[294,7547],[205,7670],[112,7489],[457,7444],[158,7244],[290,7451],[84,7824],[129,7654],[288,7454],[89,7383],[423,7635],[421,7559],[407,7244],[179,7426],[206,7611],[88,7623],[364,7547],[522,7222],[428,7981],[149,7456],[464,7533],[339,7546],[153,7631],[467,7342],[55,7890],[247,7614],[644,7684],[765,7258],[147,7665],[741,7521],[115,7207],[7,7564],[766,7901],[447,7763],[499,7242],[436,7792],[2,7526],[473,7277],[791,7272],[182,7285],[714,7735],[93,7592],[464,7904],[613,7407],[184,7446],[373,7392],[588,7435],[778,7912],[236,7217],[644,7579],[279,7742],[288,7951],[596,7917],[327,7283],[428,7538],[484,7257],[452,7795],[303,7392],[148,7310],[723,7740],[232,7232],[151,7477],[180,7605],[699,7471],[422,7967],[541,7542],[643,7900],[338,7956],[113,7787],[25,7346],[738,7235],[311,7270],[793,7868],[66,7530],[727,7512],[557,7716],[33,7997],[323,7300],[51,7821],[226,7394],[291,7600],[793,7573],[740,7453],[229,7626],[207,7982],[243,7206],[323,7693],[725,7990],[377,7916],[309,7398],[476,7917],[54,7202],[591,7753],[288,7322],[415,7989],[269,7438],[461,7659],[33,7855],[662,7350],[98,7673],[10,7964],[276,7651],[62,7313],[188,7549],[720,7916],[637,7373],[565,7896],[229,7417],[642,7778],[762,7681],[94,7659],[657,7752],[383,7589],[541,7311],[464,7587],[619,7655],[383,7237],[363,7818],[96,7705],[757,7442],[705,7992],[576,7444],[31,7981],[718,7270],[58,7902],[355,7938],[371,7803],[85,7993],[387,7821],[787,7761],[193,7453],[734,7866],[75,7699],[602,7416],[384,7494],[659,7711],[28,7731],[616,7930],[263,7940],[307,7244],[394,7558],[721,7934],[644,7503],[363,7352],[757,7542],[584,7407],[377,7678],[21,7879],[509,7591],[377,7206],[590,7467],[401,7384],[12,7680],[550,7863],[328,7635],[150,7868],[738,7386],[460,7313],[594,7877],[163,7620],[26,7978],[131,7846],[673,7592],[757,7266],[124,7774],[446,7724],[20,7957],[120,7227],[620,7425],[446,7203],[601,7864],[69,7803],[285,7785],[3,7769],[68,7869],[478,7250],[50,7401],[139,7711],[706,7615],[682,7372],[540,7224],[352,7884],[682,7906],[536,7672],[476,7702],[507,7250],[86,7484],[687,7802],[475,7337],[714,7594],[518,7332],[102,7307],[576,7543],[437,7863],[671,7991],[752,7814],[410,7513],[444,7228],[502,7660],[207,7850],[325,7808],[370,7599],[712,7776],[702,7314],[607,7371],[152,7237],[362,7323],[209,7577],[410,7445],[746,7782],[756,7878],[470,7489],[748,7743],[686,7824],[178,7560],[45,7249],[507,7565],[108,7634],[478,7964],[272,7939],[537,7626],[654,7684],[193,7571],[554,7323],[564,7333],[748,7315],[501,7944],[508,7468],[521,7660],[288,7711],[461,7821],[244,7582],[628,7837],[682,7894],[341,7870],[563,7514],[221,7749],[240,7581],[395,7818],[592,7804],[480,7969],[127,7381],[359,7348],[791,7236],[82,7826],[204,7263],[713,7589],[260,7636],[655,7791],[126,7352],[390,7702],[114,7791],[91,7782],[430,7799],[589,7393],[215,7578],[244,7894],[38,7417],[642,7278],[251,7983],[568,7689],[570,7421],[40,7528],[656,7602],[543,7335],[527,7955],[741,7688],[382,7437],[241,7537],[339,7382],[433,7906],[175,7254],[139,7812],[691,7483],[794,7762],[502,7459],[491,7716],[126,7316],[190,7771],[161,7612],[605,7880],[603,7401],[297,7963],[377,7325],[462,7436],[584,7369],[130,7340],[592,7426],[84,7651],[660,7671],[522,7641],[343,7208],[250,7781],[666,7483],[702,7938],[449,7582],[776,7859],[663,7256],[323,7533],[493,7909],[676,7706],[742,7694],[799,7776],[80,7669],[307,7408],[75,7328],[685,7508],[216,7509],[656,7526],[211,7558],[253,7714],[712,7278],[70,7906],[338,7706],[360,7232],[601,7344],[470,7361],[51,7833],[278,7727],[35,7950],[192,7589],[605,7918],[382,7661],[659,7560],[427,7390],[626,7462],[201,7404],[590,7302],[416,7748],[386,7671],[4,7733],[549,7522],[332,7216],[409,7786],[777,7478],[233,7992],[593,7858],[279,7472],[108,7261],[734,7269],[16,7507],[149,7698],[798,7966],[607,7685],[266,7837],[197,7557],[285,7798],[153,7317],[1,7309],[537,7272],[517,7758],[655,7829],[425,7825],[637,7856],[116,7419],[159,7980],[642,7800],[251,7578],[459,7528],[215,7741],[314,7714],[649,7279],[103,7291],[332,7933],[511,7238],[193,7506],[469,7578],[405,7254],[658,7294],[421,7493],[703,7365],[393,7391],[309,7542],[331,7490],[426,7366],[3,7382],[126,7436],[349,7926],[256,7598],[637,7926],[712,7373],[147,7879],[255,7783],[291,7204],[501,7285],[768,7655],[252,7882],[302,7434],[282,7594],[392,7263],[639,7548],[265,7683],[701,7250],[738,7877],[51,7627],[51,7455],[584,7738],[78,7252],[452,7791],[792,7566],[110,7600],[194,7545],[136,7276],[265,7847],[780,7955],[87,7957],[181,7200],[174,7369],[298,7848],[109,7699],[346,7608],[68,7592],[526,7242],[621,7545],[446,7459],[680,7287],[541,7949],[343,7476],[124,7396],[223,7890],[201,7645],[401,7568],[748,7212],[143,7730],[348,7811],[68,7536],[441,7913],[777,7527],[565,7319],[449,7862],[627,7301],[682,7582],[744,7684],[561,7374],[579,7260],[720,7537],[181,7955],[311,7806],[182,7374],[87,7798],[118,7858],[702,7305],[666,7489],[163,7784],[596,7822],[276,7715],[115,7497],[112,7491],[7,7969],[86,7421],[525,7570],[710,7920],[406,7791],[595,7394],[594,7323],[291,7372],[422,7529],[157,7205],[123,7366],[727,7231],[110,7661],[309,7243],[155,7252],[634,7718],[662,7509],[718,7514],[446,7833],[515,7725],[533,7600],[412,7572],[639,7504],[556,7275],[707,7755],[150,7674],[191,7814],[127,7453],[677,7594],[468,7884],[334,7821],[104,7626],[319,7479],[710,7854],[590,7326],[320,7395],[153,7936],[116,7773],[670,7226],[97,7300],[195,7263],[529,7522],[140,7422],[104,7567],[412,7746],[785,7374],[241,7496],[187,7756],[256,7932],[12,7944],[128,7351],[743,7952],[257,7604],[178,7602],[488,7283],[644,7448],[799,7361],[10,7342],[320,7452],[775,7459],[162,7447],[549,7634],[451,7240],[458,7685],[394,7230],[318,7991],[137,7833],[749,7402],[667,7849],[676,7876],[604,7969],[548,7705],[655,7923],[436,7925],[404,7681],[309,7253],[552,7917],[511,7698],[137,7276],[40,7659],[665,7546],[182,7499],[704,7253],[20,7446],[223,7915],[36,7579],[74,7219],[704,7889],[772,7812],[731,7704],[148,7490],[473,7340],[352,7483],[187,7642],[354,7496],[66,7647],[762,7882],[544,7806],[687,7412],[607,7382],[497,7644],[770,7941],[771,7926],[730,7439],[594,7764],[326,7916],[189,7424],[568,7873],[105,7764],[173,7723],[590,7395],[390,7863],[196,7886],[641,7370],[795,7670],[562,7866],[436,7728],[21,7689],[755,7240],[280,7384],[666,7437],[360,7667],[791,7866],[236,7280],[533,7828],[616,7576],[592,7440],[72,7282],[742,7636],[698,7792],[607,7391],[690,7282],[666,7395],[326,7455],[319,7924],[478,7937],[58,7554],[7,7680],[671,7203],[507,7690],[383,7206],[740,7771],[564,7992],[304,7779],[289,7874],[591,7824],[626,7609],[202,7554],[467,7586],[516,7626],[721,7624],[128,7987],[72,7443],[559,7211],[333,7792],[367,7274],[625,7999],[263,7765],[286,7957],[367,7484],[766,7375],[166,7579],[596,7540],[344,7303],[200,7393],[724,7235],[760,7535],[627,7268],[359,7501],[98,7795],[423,7413],[675,7905],[362,7312],[765,7920],[101,7226],[610,7620],[509,7516],[350,7309],[688,7435],[439,7434],[411,7974],[364,7951],[619,7210],[343,7951],[31,7457],[66,7703],[243,7708],[145,7344],[242,7467],[137,7629],[718,7698],[514,7677],[40,7926],[203,7894],[176,7421],[648,7758],[114,7343],[528,7282],[451,7264],[538,7837],[669,7942],[583,7968],[235,7561],[314,7890],[453,7959],[698,7967],[327,7342],[122,7272],[165,7651],[767,7524],[595,7633],[250,7975],[509,7422],[72,7289],[313,7820],[210,7444],[107,7216],[547,7715],[412,7365],[246,7255],[442,7949],[785,7542],[800,7795],[265,7889],[26,7227],[240,7745],[97,7721],[19,7449],[721,7627],[304,7760],[381,7932],[94,7893],[38,7245],[132,7269],[512,7356],[311,7277],[723,7458],[24,7757],[89,7658],[46,7337],[530,7468],[432,7264],[434,7849],[563,7681],[545,7656],[656,7847],[196,7388],[777,7265],[288,7781],[621,7787],[489,7211],[214,7880],[131,7221],[286,7392],[659,7518],[680,7707],[261,7418],[741,7409],[314,7512],[132,7306],[44,7621],[369,7267],[46,7221],[301,7604],[1,7267],[12,7844],[299,7313],[204,7324],[240,7544],[375,7541],[763,7666],[584,7510],[136,7248],[738,7683],[778,7350],[701,7897],[130,7895],[237,7811],[365,7633],[773,7909],[475,7333],[672,7287],[500,7586],[799,7278],[727,7735],[782,7814],[776,7772],[596,7767],[508,7932],[567,7664],[502,7633],[487,7953],[448,7579],[65,7422],[512,7691],[475,7872],[78,7331],[218,7821],[549,7286],[545,7787],[228,7829],[467,7858],[651,7364],[42,7484],[518,7687],[230,7239],[281,7569],[119,7801],[258,7526],[349,7957],[513,7619],[522,7286],[660,7954],[467,7941],[228,7487],[116,7751],[568,7902],[31,7710],[592,7873],[412,7997],[319,7945],[693,7241],[253,7841],[648,7309],[356,7696],[326,7722],[339,7834],[197,7782],[108,7985],[552,7852],[333,7444],[677,7600],[737,7343],[33,7570],[285,7513],[615,7395],[785,7441],[597,7387],[458,7283],[690,7777],[323,7457],[31,7620],[233,7728],[321,7630],[467,7208],[143,7255],[756,7679],[650,7276],[786,7563],[201,7669],[572,7872],[285,7390],[56,7903],[354,7567],[698,7864],[356,7936],[344,7973],[224,7889],[327,7473],[289,7977],[36,7660],[777,7728],[725,7807],[693,7623],[466,7679],[715,7715],[538,7750],[601,7368],[659,7932],[747,7338],[108,7980],[32,7958],[39,7666],[798,7343],[695,7882],[9,7483],[22,7987],[774,7833],[317,7706],[778,7269],[127,7377],[302,7612],[108,7904],[454,7575],[684,7703],[550,7909],[15,7486],[97,7223],[710,7518],[565,7637],[418,7695],[83,7734],[428,7463],[279,7352],[443,7329],[788,7499],[307,7709],[60,7912],[264,7776],[561,7624],[481,7284],[117,7216],[120,7714],[474,7677],[797,7214],[236,7448],[160,7534],[474,7995],[48,7909],[292,7875],[300,7429],[772,7229],[474,7450],[137,7515],[385,7875],[433,7991],[542,7779],[628,7694],[673,7462],[615,7895],[61,7972],[218,7854],[794,7258],[267,7433],[525,7303],[475,7652],[639,7809],[758,7712],[604,7787],[146,7467],[413,7787],[164,7490],[478,7222],[289,7740],[124,7307],[696,7396],[60,7295],[523,7636],[786,7495],[250,7339],[253,7285],[270,7565],[705,7588],[380,7299],[522,7327],[170,7674],[376,7808],[82,7686],[58,7222],[251,7477],[247,7977],[471,7477],[202,7946],[158,7562],[390,7899],[480,7503],[29,7905],[338,7415],[389,7341],[322,7293],[664,7887],[158,7410],[126,7274],[344,7464],[111,7852],[68,7316],[606,7227],[655,7340],[28,7588],[11,7700],[104,7453],[754,7611],[665,7797],[208,7254],[33,7569],[131,7380],[77,7891],[316,7471],[167,7862],[434,7648],[669,7450],[766,7972],[374,7404],[573,7471],[685,7339],[323,7993],[434,7322],[126,7619],[111,7901],[320,7884],[782,7836],[584,7932],[625,7518],[276,7952],[145,7623],[718,7287],[546,7481],[337,7522],[214,7649],[781,7992],[324,7748],[157,7611],[542,7330],[737,7856],[639,7577],[61,7746],[109,7476],[626,7794],[672,7518],[32,7387],[196,7336],[199,7555],[568,7579],[607,7435],[356,7723],[57,7910],[211,7885],[754,7843],[341,7860],[175,7720],[530,7230],[344,7228],[507,7523],[46,7709],[733,7881],[520,7290],[599,7902],[645,7661],[216,7569],[265,7501],[134,7566],[185,7487],[720,7918],[484,7704],[500,7694],[259,7492],[797,7480],[579,7599],[270,7801],[251,7234],[313,7637],[289,7254],[49,7645],[8,7595],[581,7744],[523,7338],[399,7504],[559,7644],[128,7959],[687,7668],[372,7573],[212,7708],[473,7807],[233,7775],[360,7867],[297,7868],[388,7221],[596,7303],[143,7324],[22,7460],[24,7679],[160,7569],[259,7562],[748,7402],[617,7645],[44,7294],[773,7644],[61,7787],[629,7427],[144,7847],[761,7458],[479,7231],[450,7777],[146,7606],[139,7569],[684,7306],[723,7688],[303,7406],[127,7902],[403,7992],[58,7422],[134,7580],[60,7864],[236,7967],[696,7909],[795,7798],[304,7452],[690,7849],[611,7720],[543,7464],[273,7358],[470,7770],[586,7566],[159,7902],[248,7637],[92,7711],[605,7470],[583,7624],[743,7468],[344,7513],[619,7915],[766,7562],[728,7838],[362,7667],[750,7719],[118,7377],[459,7942],[197,7943],[111,7704],[726,7494],[697,7945],[674,7347],[449,7946],[109,7741],[234,7335],[148,7899],[286,7224],[165,7418],[614,7413],[403,7881],[132,7687],[396,7304],[472,7618],[277,7927],[31,7984],[682,7483],[41,7484],[745,7598],[344,7879],[375,7301],[82,7367],[565,7586],[236,7716],[81,7325],[16,7558],[6,7339],[724,7334],[544,7935],[617,7365],[38,7688],[793,7865],[298,7895],[651,7546],[601,7565],[173,7738],[735,7338],[452,7998],[430,7702],[493,7479],[186,7411],[639,7415],[325,7915],[271,7422],[770,7434],[654,7667],[198,7948],[136,7932],[451,7732],[221,7822],[428,7988],[764,7792],[191,7281],[352,7322],[42,7212],[141,7588],[1,7701],[451,7974],[169,7269],[263,7398],[643,7841],[257,7913],[534,7594],[582,7882],[391,7551],[221,7708],[69,7569],[365,7687],[254,7314],[26,7564],[185,7204],[573,7781],[572,7399],[67,7769],[464,7427],[31,7881],[581,7570],[52,7687],[637,7414],[238,7678],[460,7433],[107,7333],[626,7687],[197,7481],[87,7959],[318,7729],[348,7252],[785,7363],[700,7906],[464,7885],[382,7821],[654,7913],[27,7738],[152,7621],[557,7933],[18,7200],[622,7260],[483,7354],[176,7228],[574,7870],[643,7689],[87,7360],[771,7324],[534,7798],[770,7485],[355,7384],[251,7810],[658,7916],[331,7783],[423,7990],[38,7334],[240,7414],[77,7535],[635,7998],[638,7728],[738,7675],[359,7317],[471,7569],[456,7835],[30,7402],[171,7401],[14,7764],[538,7751],[58,7644],[3,7819],[289,7268],[476,7544],[12,7373],[751,7486],[124,7426],[24,7248],[454,7725],[640,7759],[433,7705],[665,7557],[528,7917],[800,7696],[332,7548],[699,7754],[346,7205],[601,7546],[588,7971],[99,7510],[343,7244],[303,7833],[448,7380],[254,7600],[422,7836],[298,7893],[402,7331],[456,7382],[513,7639],[347,7306],[553,7327],[294,7985],[706,7337],[40,7515],[231,7866],[566,7476],[783,7878],[331,7247],[698,7905],[265,7610],[149,7308],[498,7595],[267,7878],[198,7502],[619,7406],[587,7493],[109,7635],[732,7970],[386,7659],[262,7234],[452,7769],[695,7773],[688,7911],[429,7777],[523,7953],[713,7568],[545,7522],[457,7477],[635,7277],[769,7288],[55,7604],[661,7647],[468,7856],[154,7639],[296,7293],[516,7658],[639,7441],[668,7792],[460,7750],[625,7524],[221,7665],[484,7253],[74,7363],[285,7788],[217,7312],[441,7958],[272,7564],[692,7886],[611,7734],[150,7464],[252,7649],[559,7214],[276,7650],[2,7499],[40,7694],[643,7803],[205,7598],[352,7269],[521,7657],[382,7603],[593,7382],[442,7913],[603,7342],[775,7251],[792,7996],[225,7884],[742,7362],[480,7413],[350,7994],[508,7320],[144,7572],[398,7695],[189,7274],[158,7455],[4,7227],[565,7999],[349,7668],[675,7718],[589,7557],[601,7317],[376,7439],[145,7922],[312,7385],[348,7994],[397,7589],[20,7207],[594,7752],[741,7200],[341,7702],[550,7417],[417,7869],[620,7247],[714,7410],[31,7407],[275,7274],[251,7852],[415,7546],[669,7393],[128,7802],[527,7530],[444,7949],[548,7473],[631,7895],[616,7654],[737,7651],[28,7954],[510,7244],[436,7245],[25,7710],[278,7586],[101,7484],[245,7453],[455,7232],[52,7484],[750,7805],[81,7947],[676,7699],[316,7490],[201,7596],[573,7648],[84,7876],[745,7852],[390,7444],[515,7919],[469,7647],[190,7281],[366,7689],[724,7701],[308,7206],[756,7532],[261,7583],[513,7647],[439,7324],[536,7352],[297,7978],[552,7941],[511,7514],[68,7480],[174,7700],[518,7570],[239,7842],[65,7456],[111,7943],[751,7818],[515,7712],[481,7377],[428,7412],[725,7464],[702,7785],[176,7284],[648,7364],[139,7409],[730,7424],[590,7799],[536,7363],[125,7521],[576,7224],[14,7672],[715,7325],[216,7341],[677,7912],[6,7496],[735,7479],[523,7464],[799,7559],[429,7664],[494,7272],[372,7882],[272,7313],[751,7675],[665,7912],[388,7599],[7,7503],[323,7627],[561,7803],[236,7790],[86,7301],[411,7902],[756,7517],[379,7650],[668,7610],[503,7883],[460,7969],[409,7581],[485,7936],[443,7594],[410,7457],[492,7603],[47,7775],[361,7328],[8,7712],[687,7955],[350,7856],[143,7800],[446,7518],[337,7528],[189,7272],[502,7572],[63,7494],[586,7838],[550,7731],[506,7741],[137,7807],[571,7713],[499,7347],[479,7569],[306,7958],[282,7345],[170,7205],[441,7278],[703,7916],[567,7570],[708,7241],[424,7821],[174,7809],[549,7678],[695,7437],[660,7738],[442,7902],[282,7235],[63,7589],[490,7343],[312,7624],[442,7407],[598,7238],[251,7366],[89,7463],[253,7256],[198,7291],[557,7913],[194,7780],[244,7447],[20,7280],[111,7314],[520,7469],[684,7221],[410,7562],[257,7713],[544,7531],[351,7735],[510,7527],[380,7336],[713,7496],[364,7357],[583,7602],[106,7551],[76,7483],[197,7621],[623,7668],[259,7554],[465,7924],[535,7416],[324,7274],[463,7785],[559,7404],[665,7713],[467,7929],[91,7872],[247,7536],[357,7606],[434,7832],[442,7453],[672,7408],[555,7929],[588,7563],[727,7962],[550,7647],[680,7456],[69,7675],[669,7746],[144,7534],[698,7596],[47,7909],[125,7814],[629,7967],[597,7313],[793,7952],[442,7332],[270,7634],[617,7513],[649,7681],[431,7920],[172,7806],[200,7602],[700,7749],[290,7340],[441,7737],[370,7319],[496,7804],[468,7395],[444,7692],[578,7944],[620,7249],[92,7204],[134,7870],[506,7278],[621,7977],[59,7895],[635,7520],[737,7958],[692,7330],[9,7392],[552,7876],[465,7362],[103,7331],[127,7994],[374,7852],[77,7785],[42,7849],[280,7560],[748,7591],[175,7512],[740,7477],[427,7905],[187,7354],[36,7296],[400,7544],[435,7617],[222,7847],[241,7816],[217,7894],[361,7237],[534,7359],[294,7393],[749,7334],[392,7331],[330,7446],[689,7751],[93,7483],[14,7440],[54,7858],[506,7521],[320,7809],[774,7739],[290,7267],[688,7480],[163,7262],[144,7489],[541,7504],[194,7971],[354,7250],[499,7486],[377,7214],[778,7963],[614,7417],[755,7452],[188,7727],[55,7798],[655,7778],[622,7676],[25,7343],[271,7476],[4,7509],[207,7288],[24,7834],[12,7679],[495,7298],[685,7770],[178,7441],[693,7536],[719,7906],[247,7332],[774,7700],[486,7844],[658,7425],[501,7646],[363,7551],[263,7651],[407,7266],[448,7330],[454,7565],[487,7430],[345,7528],[317,7770],[109,7495],[479,7299],[184,7598],[416,7638],[108,7547],[699,7452],[594,7422],[790,7790],[762,7330],[621,7472],[183,7673],[724,7252],[142,7270],[354,7260],[657,7300],[212,7460],[227,7851],[255,7783],[51,7647],[161,7683],[633,7958],[552,7827],[659,7538],[516,7385],[294,7657],[484,7393],[220,7921],[578,7271],[134,7817],[618,7244],[226,7660],[114,7905],[614,7312],[151,7652],[354,7813],[248,7780],[713,7571],[322,7673],[683,7289],[294,7385],[86,7957],[91,7631],[689,7391],[181,7234],[723,7600],[669,7938],[689,7533],[612,7290],[46,7203],[734,7482],[257,7394],[403,7457],[220,7601],[273,7497],[257,7415],[164,7966],[796,7776],[428,7802],[95,7458],[150,7360],[706,7978],[724,7207],[646,7257],[474,7844],[153,7369],[452,7607],[735,7712],[244,7526],[160,7672],[520,7248],[642,7719],[456,7758],[327,7884],[493,7308],[57,7816],[223,7392],[553,7968],[381,7423],[40,7214],[44,7802],[44,7261],[681,7633],[216,7849],[143,7550],[526,7948],[434,7860],[487,7656],[338,7490],[412,7578],[564,7623],[123,7690],[169,7766],[673,7516],[473,7869],[28,7447],[296,7480],[431,7853],[221,7425],[188,7928],[293,7587],[362,7864],[778,7886],[423,7741],[291,7989],[66,7542],[303,7966],[420,7952],[205,7540],[123,7676],[320,7706],[208,7521],[774,7269],[527,7575],[110,7946],[511,7605],[582,7890],[395,7567],[720,7779],[369,7793],[418,7535],[421,7212],[42,7855],[787,7434],[722,7424],[458,7961],[322,7357],[186,7783],[404,7954],[278,7728],[262,7852],[143,7290],[8,7777],[163,7388],[130,7533],[94,7949],[271,7546],[361,7691],[601,7849],[377,7937],[202,7664],[656,7333],[235,7696],[406,7623],[46,7479],[71,7491],[147,7639],[314,7419],[58,7389],[633,7683],[47,7758],[79,7425],[498,7358],[183,7804],[572,7346],[124,7216],[634,7997],[205,7570],[653,7975],[152,7518],[687,7394],[620,7356],[170,7821],[653,7578],[33,7289],[733,7889],[683,7512],[581,7603],[136,7770],[266,7226],[471,7642],[693,7861],[683,7805],[336,7402],[617,7255],[558,7679],[551,7417],[675,7872],[301,7969],[510,7612],[468,7903],[638,7224],[51,7648],[517,7359],[318,7510],[408,7641],[401,7586],[622,7516],[704,7775],[227,7641],[700,7204],[696,7778],[648,7516],[215,7730],[418,7717],[162,7898],[761,7564],[746,7216],[542,7534],[343,7793],[185,7542],[648,7637],[169,7784],[488,7388],[386,7589],[579,7968],[149,7415],[796,7346],[335,7658],[752,7362],[499,7450],[647,7511],[735,7622],[553,7934],[620,7928],[169,7684],[56,7945],[698,7476],[478,7381],[197,7999],[688,7896],[246,7459],[653,7457],[440,7626],[407,7307],[47,7561],[102,7505],[613,7946],[761,7595],[89,7759],[49,7587],[525,7605],[172,7714],[562,7595],[775,7956],[764,7732],[712,7734],[482,7830],[173,7478],[256,7818],[260,7219],[772,7451],[696,7977],[665,7892],[637,7568],[85,7732],[437,7836],[724,7531],[427,7288],[235,7883],[306,7479],[48,7774],[553,7433],[712,7871],[240,7303],[557,7453],[557,7228],[124,7604],[593,7958],[195,7474],[480,7630],[599,7269],[138,7860],[208,7705],[505,7816],[253,7846],[624,7764],[2,7293],[794,7889],[161,7363],[28,7278],[394,7420],[98,7713],[782,7575],[184,7231],[6,7842],[156,7446],[593,7301],[372,7355],[221,7274],[199,7448],[404,7852],[778,7902],[222,7895],[68,7498],[786,7204],[607,7766],[181,7281],[171,7696],[46,7387],[250,7325],[562,7809],[436,7695],[191,7235],[115,7387],[637,7556],[618,7218],[250,7300],[679,7991],[467,7697],[650,7527],[408,7789],[714,7905],[14,7207],[715,7205],[706,7307],[671,7795],[654,7969],[766,7281],[474,7585],[350,7417],[775,7378],[301,7516],[542,7444],[263,7423],[522,7996],[492,7316],[381,7274],[320,7491],[736,7704],[311,7645],[633,7318],[116,7676],[641,7259],[732,7906],[565,7663],[11,7336],[494,7344],[545,7390],[33,7460],[54,7247],[638,7443],[672,7950],[736,7732],[9,7758],[744,7406],[289,7842],[730,7781],[656,7316],[183,7409],[171,7315],[389,7882],[697,7212],[25,7583],[614,7927],[219,7382],[271,7914],[77,7759],[239,7342],[699,7953],[772,7558],[191,7928],[461,7593],[164,7569],[151,7871],[43,7367],[90,7425],[114,7301],[447,7678],[577,7554],[742,7497],[169,7741],[762,7420],[171,7595],[505,7895],[27,7303],[767,7975],[85,7610],[456,7717],[113,7315],[443,7679],[759,7749],[421,7954],[138,7812],[7,7383],[327,7298],[341,7393],[519,7606],[262,7972],[173,7642],[453,7439],[798,7216],[174,7375],[41,7829],[587,7506],[85,7694],[702,7546],[213,7856],[23,7489],[689,7212],[369,7402],[79,7822],[534,7487],[798,7276],[62,7880],[350,7228],[458,7530],[327,7202],[540,7465],[513,7527],[167,7760],[298,7622],[501,7657],[63,7438],[800,7708],[516,7456],[446,7662],[624,7703],[50,7951],[7,7772],[670,7943],[232,7224],[666,7660],[402,7479],[675,7501],[41,7547],[522,7437],[329,7535],[735,7207],[86,7709],[167,7652],[171,7665],[656,7809],[453,7448],[742,7340],[561,7391],[578,7226],[120,7906],[560,7441],[133,7211],[369,7310],[92,7969],[161,7264],[197,7461],[28,7637],[200,7592],[546,7231],[74,7832],[610,7240],[372,7230],[183,7510],[42,7610],[322,7265],[67,7890],[480,7853],[730,7479],[486,7279],[736,7290],[471,7250],[684,7602],[278,7753],[360,7650],[580,7646],[522,7769],[415,7399],[645,7873],[91,7367],[260,7257],[458,7955],[676,7818],[25,7515],[649,7504],[556,7901],[82,7452],[555,7873],[533,7323],[591,7263],[127,7302],[771,7480],[343,7915],[416,7711],[711,7611],[475,7649],[426,7272],[598,7546],[132,7474],[578,7608],[755,7229],[98,7695],[39,7781],[156,7296],[460,7632],[313,7539],[76,7694],[477,7976],[41,7332],[156,7881],[210,7768],[4,7457],[720,7558],[235,7512],[462,7716],[166,7650],[471,7872],[139,7530],[194,7946],[152,7260],[632,7561],[651,7802],[659,7906],[186,7744],[274,7736],[577,7768],[485,7441],[369,7925],[514,7710],[178,7669],[175,7834],[622,7394],[220,7843],[742,7547],[717,7683],[700,7842],[16,7367],[716,7461],[30,7827],[248,7789],[515,7610],[681,7633],[200,7931],[620,7741],[315,7984],[393,7493],[546,7492],[272,7338],[693,7665],[254,7799],[570,7438],[326,7940],[777,7602],[693,7555],[134,7927],[245,7817],[79,7766],[750,7766],[426,7877],[464,7614],[170,7359],[484,7454],[341,7431],[240,7519],[3,7599],[645,7361],[57,7306],[420,7634],[199,7451],[276,7998],[60,7319],[268,7907],[136,7294],[383,7607],[706,7400],[261,7905],[351,7381],[425,7416],[410,7375],[540,7769],[429,7386],[222,7399],[373,7670],[32,7490],[282,7472],[372,7571],[733,7320],[757,7868],[313,7408],[277,7497],[150,7286],[79,7307],[478,7547],[271,7789],[271,7500],[581,7540],[489,7856],[295,7370],[783,7282],[646,7767],[544,7935],[306,7383],[640,7986],[307,7587],[243,7717],[112,7575],[76,7365],[639,7313],[125,7617],[279,7970],[502,7857],[625,7377],[257,7876],[396,7495],[182,7800],[610,7713],[452,7377],[259,7651],[745,7409],[330,7419],[667,7836],[633,7740],[507,7583],[783,7860],[583,7321],[368,7445],[2,7262],[475,7523],[122,7467],[377,7866],[77,7741],[225,7484],[491,7782],[181,7903],[240,7249],[347,7676],[571,7375],[266,7532],[757,7401],[78,7226],[741,7715],[93,7343],[257,7614],[307,7444],[760,7689],[547,7855],[696,7593],[172,7515],[210,7962],[48,7850],[72,7475],[647,7625],[13,7635],[430,7954],[761,7804],[398,7316],[742,7227],[386,7809],[641,7623],[248,7967],[456,7516],[621,7304],[150,7265],[531,7702],[612,7916],[643,7237],[395,7546],[9,7634],[79,7991],[272,7648],[85,7558],[595,7209],[104,7310],[168,7526],[2,7342],[744,7395],[83,7625],[45,7201],[260,7979],[53,7391],[492,7369],[571,7835],[554,7625],[523,7432],[654,7336],[115,7376],[61,7767],[400,7281],[270,7965],[718,7831],[561,7502],[144,7580],[37,7673],[633,7604],[781,7589],[632,7819],[651,7970],[463,7717],[267,7367],[33,7263],[662,7269],[292,7286],[284,7476],[139,7463],[401,7214],[657,7687],[228,7700],[173,7452],[722,7746],[433,7269],[735,7562],[592,7577],[173,7731],[721,7529],[757,7813],[725,7814],[279,7667],[331,7875],[481,7914],[447,7418],[771,7701],[719,7983],[623,7715],[174,7807],[235,7229],[348,7521],[510,7689],[588,7680],[675,7815],[656,7747],[213,7599],[454,7644],[16,7680],[319,7937],[640,7302],[41,7577],[600,7423],[619,7479],[747,7888],[548,7305],[324,7204],[790,7487],[235,7783],[57,7442],[349,7644],[759,7226],[136,7743],[364,7985],[564,7847],[419,7833],[647,7592],[435,7933],[15,7832],[429,7416],[365,7514],[692,7679],[402,7217],[497,7534],[83,7375],[89,7258],[90,7712],[297,7332],[398,7369],[679,7340],[388,7518],[671,7492],[106,7739],[23,7333],[284,7553],[596,7500],[62,7713],[357,7821],[648,7439],[329,7615],[382,7446],[344,7361],[283,7856],[250,7681],[791,7664],[782,7947],[604,7538],[38,7657],[740,7627],[698,7571],[792,7992],[330,7669],[228,7697],[258,7773],[767,7944],[361,7944],[767,7358],[338,7277],[174,7374],[338,7962],[564,7497],[429,7668],[606,7855],[52,7284],[675,7703],[332,7991],[173,7361],[491,7803],[769,7849],[179,7836],[280,7776],[702,7979],[438,7535],[165,7414],[639,7606],[508,7460],[694,7798],[577,7831],[54,7345],[61,7429],[680,7729],[347,7475],[38,7648],[0,7260],[573,7468],[777,7291],[350,7784],[731,7284],[131,7489],[146,7624],[577,7957],[327,7681],[327,7426],[711,7756],[348,7270],[724,7406],[691,7822],[493,7584],[722,7453],[341,7364],[629,7491],[427,7272],[319,7333],[740,7644],[765,7371],[22,7477],[60,7723],[362,7344],[22,7607],[251,7409],[269,7617],[78,7270],[539,7813],[510,7738],[507,7393],[459,7347],[533,7383],[666,7591],[357,7817],[193,7874],[223,7908],[519,7242],[293,7304],[728,7827],[67,7692],[87,7593],[49,7451],[207,7955],[693,7436],[62,7716],[382,7498],[288,7288],[541,7865],[308,7455],[429,7868],[440,7255],[680,7385],[693,7216],[645,7819],[203,7289],[227,7682],[566,7753],[49,7948],[162,7900],[68,7232],[594,7446],[425,7697],[742,7940],[423,7710],[468,7674],[465,7445],[426,7550],[783,7667],[386,7518],[365,7556],[305,7538],[632,7259],[213,7456],[516,7313],[628,7538],[712,7456],[206,7566],[790,7956],[739,7313],[110,7251],[143,7590],[639,7409],[587,7522],[380,7410],[240,7250],[550,7833],[749,7969],[138,7396],[707,7455],[193,7868],[84,7410],[373,7895],[763,7271],[318,7870],[532,7522],[344,7964],[29,7675],[23,7346],[385,7456],[17,7642],[141,7632],[14,7594],[450,7488],[734,7308],[327,7847],[126,7964],[304,7513],[519,7688],[447,7360],[109,7630],[292,7775],[250,7952],[261,7876],[139,7664],[713,7792],[464,7455],[55,7378],[597,7488],[54,7647],[580,7535],[489,7372],[353,7371],[682,7234],[161,7434],[758,7993],[369,7719],[634,7411],[464,7673],[786,7314],[164,7583],[559,7385],[549,7470],[543,7984],[572,7817],[86,7805],[685,7935],[603,7507],[617,7566],[84,7593],[694,7942],[341,7839],[596,7268],[767,7923],[717,7797],[282,7521],[403,7872],[300,7505],[442,7457],[785,7529],[147,7443],[598,7206],[30,7464],[74,7579],[173,7510],[159,7955],[538,7584],[507,7751],[374,7538],[261,7705],[9,7270],[514,7802],[481,7980],[499,7813],[703,7554],[782,7246],[127,7901],[308,7584],[462,7244],[766,7830],[695,7971],[233,7633],[501,7346],[776,7857],[351,7245],[760,7331],[515,7914],[28,7258],[707,7517],[682,7278],[472,7695],[728,7233],[650,7707],[247,7476],[717,7633],[43,7889],[254,7308],[423,7300],[226,7879],[305,7212],[764,7699],[601,7832],[130,7901],[763,7289],[394,7474],[358,7692],[331,7381],[499,7553],[295,7863],[183,7934],[514,7928],[682,7406],[128,7978],[575,7972],[409,7701],[703,7714],[346,7292],[363,7896],[53,7633],[400,7733],[520,7271],[355,7567],[760,7679],[112,7909],[332,7573],[425,7893],[426,7282],[431,7624],[573,7796],[308,7244],[579,7445],[430,7964],[244,7597],[256,7617],[654,7214],[349,7782],[379,7521],[122,7912],[423,7319],[73,7424],[212,7801],[342,7625],[192,7391],[134,7467],[380,7276],[523,7352],[382,7604],[621,7613],[68,7807],[131,7283],[148,7856],[726,7639],[581,7661],[756,7519],[571,7978],[716,7597],[593,7697],[566,7511],[538,7693],[584,7971],[519,7842],[621,7618],[757,7402],[87,7785],[435,7429],[59,7555],[165,7706],[642,7252],[266,7925],[209,7598],[584,7387],[184,7980],[355,7776],[749,7280],[432,7456],[136,7440],[4,7323],[137,7694],[288,7615],[600,7935],[784,7210],[113,7699],[399,7408],[110,7971],[620,7292],[584,7714],[435,7449],[477,7898],[667,7217],[536,7571],[683,7554],[231,7597],[4,7630],[153,7377],[197,7212],[423,7393],[356,7627],[481,7411],[656,7954],[361,7702],[465,7582],[400,7835],[748,7496],[170,7984],[622,7733],[316,7266],[1,7505],[8,7926],[36,7422],[762,7363],[480,7408],[18,7423],[770,7330],[315,7985],[536,7378],[726,7655],[311,7911],[390,7411],[766,7532],[130,7899],[742,7434],[349,7623],[82,7318],[570,7727],[707,7970],[154,7213],[248,7753],[3,7903],[439,7888],[720,7937],[596,7537],[517,7882],[422,7342],[677,7268],[575,7542],[204,7865],[680,7265],[727,7305],[96,7871],[167,7714],[780,7280],[477,7696],[617,7230],[214,7852],[400,7634],[394,7602],[234,7601],[666,7937],[631,7511],[491,7379],[519,7275],[665,7452],[162,7624],[334,7795],[527,7486],[344,7210],[197,7405],[700,7703],[625,7250],[734,7377],[683,7879],[615,7484],[31,7430],[502,7569],[223,7553],[648,7905],[202,7882],[760,7310],[675,7405],[159,7927],[256,7957],[351,7329],[705,7658],[705,7537],[424,7662],[100,7796],[646,7550],[48,7949],[255,7255],[378,7612],[513,7665],[568,7263],[467,7268],[128,7917],[791,7607],[337,7355],[290,7837],[299,7754],[618,7273],[512,7564],[163,7940],[613,7717],[243,7719],[514,7373],[4,7930],[75,7535],[123,7483],[601,7678],[452,7708],[423,7474],[588,7291],[251,7609],[743,7286],[418,7560],[451,7623],[354,7340],[520,7370],[137,7920],[768,7596],[111,7781],[671,7535],[672,7475],[725,7777],[682,7224],[134,7776],[312,7775],[269,7794],[797,7503],[194,7887],[762,7462],[758,7527],[469,7691],[419,7604],[78,7939],[243,7755],[548,7647],[351,7840],[348,7216],[795,7954],[34,7924],[596,7292],[714,7576],[608,7554],[737,7609],[762,7654],[527,7392],[145,7413],[577,7913],[666,7959],[782,7573],[774,7805],[774,7392],[392,7215],[119,7793],[798,7491],[89,7357],[652,7670],[31,7625],[392,7308],[575,7993],[335,7887],[235,7619],[35,7922],[543,7704],[547,7242],[448,7339],[0,7884],[143,7229],[228,7849],[729,7419],[563,7762],[102,7268],[161,7211],[776,7791],[632,7637],[405,7688],[438,7425],[453,7367],[17,7404],[662,7654],[228,7502],[792,7456],[515,7429],[384,7570],[430,7694],[0,7671],[655,7303],[593,7844],[713,7202],[355,7615],[675,7533],[502,7493],[198,7803],[417,7700],[489,7234],[325,7466],[400,7661],[206,7815],[249,7277],[401,7457],[743,7811],[5,7804],[760,7549],[277,7635],[562,7869],[545,7707],[764,7446],[735,7876],[51,7973],[362,7729],[514,7639],[546,7327],[59,7364],[338,7417],[96,7339],[603,7380],[432,7629],[130,7523],[278,7707],[796,7720],[49,7902],[33,7777],[335,7728],[52,7513],[14,7292],[798,7500],[395,7325],[323,7822],[478,7987],[750,7213],[41,7226],[72,7793],[687,7256],[35,7661],[214,7731],[518,7527],[263,7601],[451,7976],[726,7437],[645,7516],[745,7708],[289,7449],[411,7971],[586,7437],[587,7851],[33,7767],[618,7482],[107,7415],[109,7788],[543,7525],[589,7587],[509,7269],[536,7424],[146,7311],[205,7213],[358,7500],[128,7876],[554,7852],[259,7870],[726,7379],[367,7845],[69,7306],[88,7796],[671,7242],[201,7973],[361,7815],[307,7965],[14,7791],[300,7370],[677,7791],[289,7264],[185,7521],[578,7659],[636,7839],[756,7803],[37,7946],[699,7433],[778,7990],[142,7452],[571,7755],[726,7658],[695,7628],[479,7598],[440,7237],[647,7596],[169,7945],[536,7791],[327,7272],[206,7999],[317,7217],[663,7417],[577,7826],[481,7763],[223,7746],[472,7493],[297,7885],[716,7868],[199,7221],[540,7302],[522,7401],[417,7444],[164,7907],[774,7782],[680,7409],[648,7838],[700,7658],[722,7208],[184,7594],[770,7591],[248,7722],[170,7204],[483,7325],[356,7516],[572,7281],[564,7533],[120,7427],[228,7307],[79,7681],[521,7624],[57,7549],[448,7659],[415,7663],[505,7273],[607,7339],[721,7880],[572,7765],[621,7338],[781,7234],[233,7260],[159,7743],[454,7527],[448,7476],[34,7566],[515,7839],[770,7390],[243,7895],[13,7414],[427,7213],[387,7648],[500,7742],[493,7313],[608,7858],[173,7472],[777,7811],[483,7878],[179,7587],[331,7549],[768,7676],[65,7393],[414,7488],[590,7624],[488,7464],[714,7572],[97,7911],[762,7516],[323,7434],[463,7564],[783,7381],[363,7654],[736,7507],[304,7894],[530,7744],[149,7530],[142,7937],[429,7870],[618,7787],[49,7924],[115,7733],[374,7306],[780,7305],[114,7522],[753,7643],[487,7455],[237,7756],[352,7276],[327,7670],[438,7493],[753,7630],[506,7237],[664,7577],[61,7876],[423,7968],[544,7209],[623,7757],[207,7716],[752,7414],[69,7796],[228,7465],[444,7531],[719,7261],[181,7977],[496,7527],[521,7638],[52,7454],[640,7506],[764,7963],[352,7585],[207,7904],[246,7411],[796,7753],[591,7354],[250,7507],[638,7568],[141,7485],[787,7528],[462,7254],[347,7278],[324,7545],[739,7681],[543,7853],[136,7612],[196,7748],[556,7207],[655,7407],[170,7475],[693,7956],[148,7675],[90,7586],[112,7881],[537,7965],[687,7599],[791,7364],[732,7904],[523,7847],[386,7793],[473,7953],[256,7692],[427,7940],[184,7420],[647,7680],[25,7414],[700,7769],[352,7867],[285,7680],[573,7975],[454,7970],[711,7389],[423,7739],[98,7784],[193,7325],[625,7358],[183,7941],[585,7487],[92,7990],[729,7379],[718,7428],[142,7722],[771,7913],[0,7982],[401,7389],[356,7446],[701,7793],[256,7949],[97,7713],[369,7481],[349,7576],[284,7265],[552,7827],[168,7525],[551,7742],[244,7287],[267,7984],[3,7252],[237,7555],[411,7430],[28,7407],[162,7717],[154,7641],[486,7395],[334,7883],[310,7827],[9,7515],[457,7793],[773,7759],[389,7568],[453,7302],[436,7715],[598,7930],[447,7869],[142,7294],[199,7351],[798,7956],[375,7261],[537,7627],[176,7752],[417,7749],[321,7595],[299,7840],[215,7505],[448,7595],[51,7901],[696,7736],[551,7379],[215,7220],[590,7374],[370,7757],[449,7381],[533,7812],[666,7855],[296,7604],[478,7880],[218,7602],[214,7279],[406,7407],[249,7216],[567,7500],[396,7242],[105,7680],[161,7962],[403,7355],[526,7987],[14,7441],[212,7917],[414,7519],[623,7250],[527,7490],[683,7711],[288,7784],[82,7846],[92,7643],[245,7460],[667,7924],[227,7896],[35,7434],[97,7262],[730,7906],[100,7597],[421,7440],[787,7584],[441,7383],[769,7347],[536,7480],[371,7525],[313,7274],[297,7968],[2,7746],[650,7539],[464,7830],[387,7355],[166,7788],[360,7277],[782,7306],[649,7201],[523,7519],[539,7637],[81,7237],[391,7948],[792,7624],[765,7244],[757,7875],[61,7387],[398,7206],[303,7628],[605,7218],[652,7306],[452,7468],[627,7875],[343,7404],[523,7849],[639,7308],[736,7588],[137,7519],[284,7274],[24,7626],[398,7841],[503,7758],[557,7494],[327,7681],[226,7351],[77,7640],[357,7257],[181,7899],[280,7356],[216,7992],[537,7785],[199,7554],[272,7960],[260,7214],[471,7278],[69,7857],[15,7903],[475,7974],[175,7831],[528,7583],[24,7372],[249,7877],[674,7226],[354,7854],[767,7231],[435,7644],[455,7296],[258,7753],[347,7439],[227,7872],[203,7994],[124,7711],[782,7282],[260,7666],[588,7978],[324,7565],[261,7953],[67,7374],[576,7765],[711,7369],[299,7971],[776,7792],[182,7360],[5,7384],[519,7887],[562,7273],[365,7869],[185,7248],[521,7800],[696,7562],[142,7313],[451,7601],[560,7203],[123,7361],[427,7821],[440,7370],[342,7246],[335,7977],[58,7688],[754,7433],[729,7339],[285,7967],[614,7264],[237,7815],[302,7444],[55,7452],[704,7968],[172,7574],[46,7959],[773,7372],[498,7456],[614,7606],[155,7445],[574,7435],[517,7488],[302,7248],[776,7895],[237,7693],[657,7602],[363,7463],[215,7810],[598,7806],[272,7358],[748,7775],[775,7564],[755,7737],[19,7395],[230,7226],[535,7620],[655,7460],[98,7494],[569,7621],[671,7451],[131,7804],[664,7247],[527,7526],[20,7540],[58,7972],[278,7887],[523,7707],[676,7308],[342,7798],[28,7522],[120,7770],[128,7416],[477,7699],[174,7307],[60,7829],[53,7694],[259,7495],[263,7681],[586,7907],[237,7781],[109,7908],[73,7682],[127,7696],[257,7931],[10,7230],[205,7927],[262,7565],[403,7740],[695,7716],[704,7483],[542,7605],[337,7890],[498,7972],[213,7690],[300,7999],[284,7813],[387,7614],[386,7205],[453,7338],[224,7603],[387,7947],[59,7985],[299,7681],[269,7306],[151,7206],[571,7308],[556,7201],[197,7646],[529,7808],[215,7713],[654,7357],[364,7335],[546,7894],[599,7840],[288,7204],[313,7421],[594,7452],[792,7573],[119,7611],[18,7340],[405,7822],[792,7925],[336,7694],[451,7582],[135,7906],[440,7628],[574,7496],[572,7252],[222,7882],[758,7301],[437,7401],[409,7681],[535,7271],[470,7522],[153,7467],[663,7766],[68,7454],[346,7449],[499,7594],[274,7360],[624,7727],[235,7974],[627,7425],[11,7414],[639,7981],[366,7446],[613,7575],[138,7958],[714,7606],[42,7734],[669,7750],[354,7392],[185,7814],[572,7679],[242,7497],[721,7230],[95,7286],[260,7834],[616,7700],[737,7511],[466,7536],[278,7944],[501,7406],[688,7869],[460,7674],[37,7482],[779,7802],[630,7240],[589,7627],[469,7746],[69,7817],[317,7573],[398,7544],[588,7690],[461,7438],[624,7594],[96,7998],[32,7879],[205,7991],[703,7456],[21,7551],[481,7410],[411,7754],[746,7794],[98,7252],[531,7963],[20,7500],[112,7922],[690,7708],[321,7639],[639,7854],[284,7419],[32,7685],[730,7365],[106,7901],[514,7876],[518,7667],[677,7664],[69,7893],[736,7634],[590,7802],[103,7596],[327,7718],[473,7385],[237,7620],[253,7515],[781,7271],[121,7232],[270,7750],[339,7650],[569,7931],[429,7660],[113,7348],[394,7535],[488,7910],[163,7762],[68,7766],[147,7988],[600,7320],[170,7702],[776,7775],[638,7856],[546,7721],[375,7864],[168,7464],[591,7889],[596,7265],[639,7614],[26,7853],[542,7499],[102,7936],[712,7556],[693,7773],[797,7542],[376,7397],[476,7930],[232,7263],[723,7484],[219,7666],[426,7408],[126,7742],[328,7577],[579,7613],[505,7222],[504,7326],[445,7969],[637,7461],[261,7760],[727,7297],[277,7967],[312,7202],[4,7260],[764,7951],[278,7237],[455,7578],[420,7246],[152,7444],[792,7325],[52,7944],[432,7519],[794,7241],[455,7987],[621,7697],[162,7798],[420,7736],[758,7365],[180,7950],[575,7715],[677,7538],[767,7512],[157,7664],[211,7960],[273,7464],[602,7928],[497,7789],[758,7813],[152,7583],[117,7857],[610,7691],[706,7399],[420,7617],[646,7529],[282,7656],[750,7575],[520,7672],[573,7919],[683,7944],[249,7882],[495,7268],[61,7917],[330,7392],[601,7962],[712,7346],[363,7863],[731,7804],[56,7640],[709,7707],[470,7323],[571,7531],[199,7931],[410,7202],[786,7850],[109,7686],[296,7429],[717,7220],[263,7262],[161,7558],[386,7260],[47,7500],[123,7405],[101,7618],[698,7606],[506,7676],[360,7351],[491,7995],[0,7648],[113,7359],[458,7789],[723,7952],[705,7679],[105,7649],[440,7418],[759,7533],[9,7620],[779,7502],[182,7913],[469,7361],[583,7937],[622,7224],[704,7501],[215,7724],[560,7844],[103,7698],[736,7616],[472,7607],[138,7371],[448,7235],[147,7955],[356,7519],[252,7273],[405,7855],[237,7788],[68,7526],[522,7516],[431,7479],[494,7329],[343,7628],[55,7489],[393,7678],[524,7361],[669,7245],[698,7409],[446,7835],[21,7600],[710,7310],[327,7927],[719,7559],[33,7879],[189,7602],[66,7443],[263,7230],[718,7838],[482,7431],[95,7338],[158,7989],[688,7256],[729,7642],[393,7649],[445,7615],[330,7493],[671,7204],[332,7439],[753,7850],[183,7484],[160,7842],[667,7316],[708,7621],[578,7484],[226,7542],[229,7918],[170,7712],[763,7644],[75,7677],[563,7881],[12,7574],[567,7269],[245,7731],[32,7540],[463,7429],[191,7318],[228,7587],[794,7654],[571,7975],[669,7773],[394,7503],[365,7540],[381,7336],[418,7312],[322,7322],[30,7651],[11,7952],[356,7881],[374,7709],[8,7572],[160,7837],[382,7236],[575,7608],[544,7704],[273,7496],[94,7329],[790,7395],[351,7889],[600,7796],[715,7759],[603,7644],[671,7377],[325,7774],[406,7811],[154,7608],[252,7521],[441,7225],[35,7592],[663,7391],[342,7881],[354,7264],[374,7289],[494,7979],[601,7973],[457,7911],[371,7277],[599,7405],[486,7733],[256,7802],[342,7423],[68,7376],[465,7402],[273,7983],[257,7542],[542,7861],[20,7885],[67,7839],[504,7633],[704,7363],[114,7402],[275,7626],[781,7291],[406,7483],[699,7585],[78,7527],[152,7916],[441,7988],[343,7454],[154,7344],[130,7891],[267,7992],[460,7240],[759,7892],[522,7297],[2,7309],[276,7740],[707,7310],[171,7796],[737,7653],[74,7830],[571,7748],[279,7244],[792,7768],[615,7874],[257,7452],[458,7986],[309,7767],[682,7982],[663,7944],[750,7322],[139,7611],[527,7312],[499,7972],[615,7455],[8,7945],[14,7733],[18,7531],[527,7754],[606,7822],[12,7537],[670,7720],[42,7759],[123,7474],[16,7397],[221,7489],[729,7225],[348,7234],[566,7958],[115,7911],[739,7249],[347,7585],[727,7426],[349,7856],[530,7937],[673,7642],[605,7804],[38,7849],[188,7308],[796,7441],[70,7299],[420,7918],[424,7782],[310,7628],[22,7340],[651,7276],[164,7497],[440,7658],[432,7994],[264,7347],[517,7317],[152,7922],[701,7941],[744,7555],[579,7736],[194,7606],[795,7362],[52,7668],[273,7836],[521,7773],[269,7798],[706,7941],[328,7813],[76,7376],[398,7204],[592,7352],[0,7450],[229,7280],[741,7465],[762,7643],[378,7627],[703,7562],[507,7932],[122,7977],[429,7600],[362,7702],[524,7846],[98,7416],[23,7492],[588,7838],[353,7258],[310,7440],[513,7783],[691,7971],[405,7497],[664,7302],[407,7477],[360,7240],[541,7594],[147,7941],[723,7777],[330,7394],[133,7987],[130,7431],[23,7683],[151,7477],[744,7765],[500,7594],[260,7296],[190,7623],[101,7422],[685,7782],[372,7997],[371,7701],[257,7528],[557,7606],[121,7716],[763,7213],[704,7530],[516,7404],[590,7451],[319,7237],[530,7424],[377,7477],[648,7485],[144,7891],[654,7685],[527,7743],[361,7778],[558,7722],[127,7803],[176,7215],[443,7826],[392,7581],[390,7827],[200,7382],[379,7880],[612,7624],[145,7669],[446,7904],[167,7342],[245,7624],[551,7441],[84,7378],[481,7548],[236,7460],[125,7745],[210,7827],[757,7821],[720,7970],[352,7547],[650,7681],[574,7966],[797,7233],[164,7337],[526,7218],[90,7665],[509,7282],[694,7227],[430,7333],[577,7503],[371,7218],[136,7966],[395,7876],[776,7785],[293,7473],[465,7428],[795,7573],[682,7418],[284,7297],[657,7488],[713,7452],[636,7274],[31,7960],[222,7390],[244,7953],[717,7326],[541,7641],[678,7966],[437,7679],[586,7852],[51,7634],[432,7855],[498,7962],[263,7681],[493,7861],[649,7339],[769,7252],[729,7299],[81,7877],[62,7502],[1,7764],[235,7721],[256,7452],[759,7679],[571,7377],[403,7218],[86,7857],[559,7657],[480,7928],[487,7835],[771,7554],[659,7427],[192,7964],[289,7345],[136,7532],[216,7805],[231,7830],[132,7337],[270,7331],[368,7982],[504,7688],[461,7988],[185,7666],[467,7458],[370,7656],[690,7547],[49,7368],[109,7874],[533,7738],[352,7391],[95,7302],[288,7738],[316,7642],[35,7432],[728,7587],[194,7743],[235,7796],[405,7928],[784,7747],[500,7755],[517,7504],[180,7216],[776,7966],[683,7492],[473,7539],[704,7840],[708,7801],[790,7773],[252,7577],[679,7715],[608,7518],[457,7970],[486,7257],[392,7790],[718,7939],[250,7898],[603,7381],[538,7548],[247,7333],[660,7854],[55,7353],[324,7517],[417,7283],[429,7929],[681,7920],[660,7685],[606,7496],[421,7964],[455,7860],[463,7737],[680,7269],[532,7979],[583,7729],[444,7420],[734,7624],[764,7280],[230,7787],[333,7833],[735,7497],[798,7852],[335,7799],[138,7921],[66,7317],[729,7773],[343,7966],[448,7377],[235,7847],[340,7845],[52,7289],[633,7487],[471,7418],[506,7699],[584,7632],[140,7610],[690,7401],[517,7797],[388,7561],[299,7984],[576,7837],[505,7895],[475,7783],[87,7924],[490,7767],[642,7846],[157,7338],[20,7245],[563,7935],[660,7326],[23,7230],[99,7960],[261,7382],[516,7617],[683,7626],[610,7259],[404,7386],[327,7287],[329,7700],[331,7800],[348,7310],[259,7796],[785,7972],[779,7243],[578,7916],[407,7655],[100,7847],[714,7886],[14,7879],[523,7208],[1,7930],[271,7846],[361,7727],[711,7695],[187,7479],[160,7874],[178,7695],[689,7856],[300,7937],[708,7803],[403,7633],[357,7920],[675,7331],[634,7504],[528,7403],[45,7553],[117,7667],[127,7800],[475,7359],[760,7539],[65,7384],[668,7422],[262,7410],[777,7440],[335,7437],[442,7674],[382,7891],[458,7896],[495,7709],[454,7417],[256,7939],[133,7597],[787,7542],[544,7693],[330,7249],[211,7881],[71,7209],[306,7925],[662,7804],[448,7813],[94,7462],[1,7681],[488,7276],[644,7902],[642,7386],[522,7230],[422,7287],[676,7790],[610,7236],[492,7470],[403,7602],[724,7622],[618,7981],[509,7884],[279,7685],[155,7967],[92,7670],[122,7968],[174,7496],[786,7435],[377,7599],[433,7712],[672,7450],[752,7251],[223,7357],[356,7975],[201,7596],[586,7580],[732,7347],[371,7533],[544,7656],[745,7953],[388,7701],[211,7527],[746,7210],[485,7773],[777,7295],[764,7256],[677,7711],[180,7735],[705,7429],[668,7809],[304,7672],[395,7474],[743,7625],[767,7894],[73,7387],[696,7587],[90,7466],[624,7914],[654,7822],[209,7872],[414,7886],[242,7309],[789,7786],[404,7380],[451,7541],[403,7914],[305,7461],[228,7794],[780,7239],[597,7526],[65,7844],[619,7544],[402,7610],[222,7216],[63,7867],[403,7405],[399,7348],[728,7320],[55,7567],[695,7279],[398,7465],[57,7656],[522,7673],[758,7929],[663,7256],[307,7713],[110,7292],[209,7958],[669,7572],[783,7971],[200,7904],[746,7687],[766,7846],[686,7846],[252,7818],[425,7573],[163,7694],[643,7367],[783,7809],[324,7794],[767,7757],[401,7651],[471,7680],[713,7262],[342,7569],[305,7729],[606,7798],[731,7988],[37,7685],[91,7386],[710,7801],[619,7593],[623,7983],[615,7298],[333,7499],[720,7762],[187,7689],[9,7250],[672,7330],[587,7453],[222,7994],[712,7307],[217,7331],[400,7721],[585,7399],[482,7468],[430,7220],[160,7265],[255,7644],[776,7758],[548,7494],[77,7568],[511,7693],[85,7909],[138,7959],[622,7269],[538,7472],[176,7917],[640,7861],[383,7264],[739,7804],[361,7623],[770,7287],[484,7646],[682,7573],[344,7771],[614,7360],[158,7425],[709,7577],[370,7332],[707,7759],[330,7769],[502,7884],[26,7647],[199,7205],[682,7354],[786,7351],[560,7996],[730,7381],[622,7428],[241,7565],[147,7756],[449,7739],[627,7709],[106,7583],[743,7398],[359,7812],[664,7619],[307,7696],[160,7620],[538,7880],[317,7955],[510,7472],[422,7483],[692,7568],[328,7234],[465,7900],[154,7732],[251,7305],[776,7210],[125,7763],[361,7943],[700,7250],[538,7759],[50,7752],[491,7523],[412,7957],[317,7559],[768,7226],[765,7893],[346,7755],[113,7866],[12,7527],[542,7210],[111,7201],[575,7857],[569,7896],[430,7698],[608,7703],[266,7459],[511,7645],[81,7802],[217,7657],[544,7990],[747,7761],[437,7585],[198,7593],[649,7502],[397,7981],[338,7448],[224,7439],[307,7654],[453,7483],[761,7787],[418,7434],[203,7691],[591,7351],[159,7909],[440,7374],[514,7317],[630,7276],[353,7200],[185,7423],[496,7497],[211,7762],[654,7710],[250,7495],[320,7251],[330,7368],[118,7814],[296,7208],[327,7848],[3,7458],[484,7333],[769,7626],[27,7839],[489,7856],[577,7637],[167,7809],[77,7476],[363,7801],[498,7661],[241,7619],[386,7216],[585,7745],[769,7959],[324,7794],[117,7821],[570,7295],[449,7969],[305,7696],[737,7437],[398,7646],[373,7409],[628,7651],[661,7741],[539,7558],[536,7661],[306,7718],[765,7809],[699,7896],[354,7870],[370,7663],[119,7450],[115,7846],[626,7748],[259,7468],[270,7940],[615,7599],[66,7645],[248,7778],[327,7385],[347,7690],[256,7688],[488,7694],[284,7442],[449,7855],[161,7281],[145,7427],[519,7480],[527,7666],[455,7578],[392,7274],[635,7283],[561,7974],[576,7731],[480,7540],[382,7704],[606,7884],[550,7426],[129,7948],[191,7808],[41,7854],[797,7229],[61,7531],[603,7212],[327,7493],[359,7957],[116,7648],[235,7588],[329,7763],[373,7491],[33,7530],[746,7780],[68,7785],[229,7483],[310,7389],[297,7618],[128,7873],[757,7680],[105,7337],[192,7824],[239,7452],[56,7993],[781,7616],[344,7442],[207,7580],[199,7930],[377,7827],[483,7521],[599,7886],[554,7220],[584,7605],[160,7715],[190,7947],[720,7322],[608,7613],[0,7737],[501,7773],[500,7982],[644,7533],[648,7681],[516,7867],[565,7349],[796,7918],[487,7394],[476,7635],[253,7248],[602,7680],[466,7424],[594,7856],[527,7942],[253,7647],[394,7621],[27,7782],[511,7873],[583,7710],[515,7533],[343,7211],[772,7256],[702,7858],[299,7484],[372,7877],[47,7654],[721,7343],[615,7609],[160,7824],[746,7991],[758,7227],[702,7283],[17,7652],[626,7436],[727,7386],[706,7351],[414,7564],[151,7659],[529,7362],[353,7496],[395,7443],[16,7540],[636,7511],[686,7280],[755,7559],[435,7711],[123,7269],[469,7652],[327,7408],[598,7764],[789,7942],[452,7368],[284,7852],[323,7766],[92,7933],[604,7766],[471,7886],[35,7718],[687,7712],[311,7207],[506,7537],[728,7522],[783,7405],[707,7763],[2,7573],[200,7959],[401,7546],[633,7841],[488,7345],[447,7734],[222,7834],[379,7234],[787,7664],[758,7914],[24,7557],[84,7394],[46,7618],[620,7625],[459,7386],[66,7998],[131,7747],[109,7525],[397,7760],[248,7766],[460,7228],[187,7695],[533,7541],[229,7660],[582,7770],[789,7359],[114,7495],[742,7787],[174,7517],[86,7428],[484,7305],[490,7447],[2,7636],[705,7396],[579,7236],[541,7580],[543,7420],[34,7617],[572,7467],[699,7713],[209,7635],[797,7981],[528,7326],[433,7653],[165,7403],[548,7604],[435,7544],[682,7623],[47,7608],[486,7692],[380,7558],[577,7564],[459,7449],[752,7971],[334,7497],[102,7203],[326,7962],[775,7854],[685,7652],[492,7874],[133,7562],[12,7500],[40,7383],[336,7904],[766,7745],[666,7829],[755,7714],[561,7266],[26,7843],[553,7527],[333,7741],[458,7354],[216,7550],[496,7715],[298,7974],[368,7683],[644,7352],[47,7739],[21,7356],[295,7316],[16,7493],[87,7578],[60,7678],[752,7981],[310,7793],[250,7319],[415,7588],[14,7735],[687,7321],[438,7984],[696,7317],[88,7551],[650,7575],[644,7591],[447,7291],[236,7863],[291,7217],[721,7482],[451,7879],[670,7564],[46,7243],[42,7519],[343,7765],[308,7234],[656,7767],[601,7341],[133,7841],[489,7223],[415,7784],[637,7898],[38,7866],[393,7517],[621,7515],[638,7447],[340,7647],[664,7556],[756,7912],[665,7718],[41,7894],[230,7871],[403,7416],[575,7422],[413,7499],[498,7343],[168,7511],[371,7533],[789,7722],[761,7299],[496,7239],[536,7947],[226,7441],[357,7266],[781,7991],[303,7858],[683,7388],[266,7478],[662,7851],[627,7845],[727,7616],[411,7919],[253,7887],[622,7236],[788,7241],[747,7235],[266,7565],[216,7805],[727,7761],[18,7479],[718,7941],[563,7978],[556,7954],[427,7707],[647,7369],[323,7306],[308,7210],[703,7789],[385,7312],[756,7648],[52,7778],[593,7714],[458,7415],[75,7709],[14,7804],[513,7318],[407,7616],[546,7774],[105,7428],[18,7435],[641,7758],[301,7588],[617,7311],[695,7384],[562,7298],[373,7255],[103,7590],[258,7533],[267,7618],[394,7357],[378,7649],[237,7314],[158,7557],[224,7631],[110,7330],[451,7612],[607,7712],[463,7604],[176,7988],[165,7974],[598,7563],[510,7963],[654,7798],[316,7700],[286,7572],[38,7768],[61,7846],[577,7863],[517,7831],[462,7322],[477,7437],[667,7425],[91,7976],[344,7992],[507,7296],[434,7237],[284,7376],[391,7284],[271,7988],[258,7658],[89,7627],[340,7237],[271,7468],[784,7307],[378,7389],[440,7467],[290,7553],[698,7875],[512,7957],[648,7346],[613,7624],[756,7810],[292,7945],[377,7463],[88,7875],[689,7787],[117,7424],[747,7674],[571,7245],[260,7809],[518,7696],[438,7338],[21,7817],[193,7760],[196,7654],[252,7808],[401,7533],[753,7368],[319,7377],[486,7886],[426,7202],[721,7697],[288,7875],[692,7323],[318,7647],[267,7981],[276,7203],[174,7645],[710,7326],[194,7560],[195,7489],[158,7533],[361,7549],[394,7288],[570,7402],[146,7640],[737,7356],[90,7412],[546,7461],[125,7904],[368,7552],[74,7209],[589,7883],[360,7479],[135,7843],[370,7991],[144,7692],[347,7206],[376,7611],[554,7715],[653,7263],[522,7480],[662,7738],[176,7913],[13,7649],[160,7682],[73,7354],[400,7392],[210,7514],[33,7908],[679,7388],[672,7833],[216,7700],[542,7307],[750,7454],[155,7616],[481,7362],[462,7453],[249,7358],[419,7328],[459,7657],[738,7639],[449,7498],[444,7829],[343,7718],[520,7326],[594,7849],[299,7980],[394,7750],[457,7855],[723,7668],[770,7501],[597,7902],[154,7788],[296,7474],[621,7825],[198,7274],[206,7627],[500,7554],[652,7706],[160,7945],[768,7612],[418,7716],[266,7926],[618,7204],[703,7538],[235,7614],[664,7632],[449,7861],[513,7414],[781,7391],[32,7395],[77,7606],[126,7930],[728,7995],[211,7309],[495,7660],[611,7215],[349,7449],[309,7242],[212,7767],[696,7746],[226,7393],[204,7304],[669,7526],[680,7945],[98,7724],[16,7307],[407,7612],[286,7753],[11,7361],[255,7633],[138,7750],[59,7322],[611,7427],[512,7741],[709,7955],[629,7859],[374,7398],[73,7312],[362,7482],[793,7884],[246,7776],[495,7757],[96,7568],[496,7451],[266,7288],[325,7793],[86,7222],[689,7586],[793,7636],[12,7712],[430,7387],[241,7932],[389,7561],[61,7496],[406,7238],[144,7620],[668,7378],[146,7518],[506,7765],[654,7819],[134,7348],[344,7307],[653,7763],[713,7349],[436,7856],[101,7994],[250,7545],[44,7625],[128,7266],[61,7742],[296,7233],[79,7530],[491,7828],[595,7708],[667,7218],[404,7339],[726,7829],[536,7265],[38,7777],[626,7689],[657,7562],[188,7632],[543,7605],[428,7583],[50,7746],[219,7628],[565,7295],[781,7600],[570,7889],[167,7316],[297,7242],[453,7431],[482,7400],[55,7738],[702,7664],[84,7487],[772,7402],[592,7218],[199,7833],[10,7817],[158,7226],[53,7549],[756,7880],[400,7381],[407,7240],[493,7207],[647,7381],[667,7491],[504,7492],[525,7534],[668,7659],[638,7735],[733,7747],[542,7830],[175,7264],[236,7372],[472,7433],[545,7320],[338,7378],[156,7810],[212,7377],[138,7788],[464,7886],[695,7378],[789,7535],[262,7238],[114,7503],[606,7886],[692,7454],[563,7900],[302,7701],[789,7576],[298,7440],[656,7287],[518,7749],[411,7518],[369,7814],[274,7665],[559,7253],[58,7686],[644,7787],[177,7954],[276,7971],[438,7421],[409,7620],[613,7735],[247,7689],[358,7568],[653,7865],[374,7863],[17,7486],[344,7757],[249,7752],[504,7569],[39,7213],[759,7711],[765,7675],[32,7831],[595,7255],[749,7392],[118,7654],[148,7297],[645,7279],[232,7984],[371,7801],[723,7712],[2,7489],[197,7414],[242,7312],[775,7583],[689,7630],[594,7630],[270,7218],[118,7249],[85,7519],[415,7231],[92,7832],[693,7404],[213,7534],[190,7207],[445,7370],[178,7592],[55,7903],[580,7412],[79,7597],[761,7482],[133,7978],[395,7608],[232,7436],[119,7588],[96,7499],[53,7762],[159,7394],[107,7207],[65,7686],[87,7907],[459,7323],[616,7796],[302,7524],[476,7956],[256,7247],[342,7980],[112,7818],[681,7852],[586,7570],[29,7767],[75,7812],[726,7211],[5,7975],[340,7622],[789,7651],[238,7852],[366,7361],[427,7431],[373,7812],[377,7289],[483,7642],[244,7570],[186,7470],[112,7445],[667,7228],[153,7921],[190,7739],[775,7585],[490,7300],[55,7606],[424,7534],[760,7633],[22,7372],[301,7795],[171,7397],[375,7656],[450,7654],[294,7283],[680,7508],[593,7244],[376,7682],[87,7710],[345,7396],[154,7331],[730,7707],[698,7839],[632,7705],[710,7893],[786,7556],[167,7603],[124,7523],[539,7675],[535,7839],[720,7736],[609,7931],[118,7610],[746,7767],[255,7301],[488,7308],[324,7316],[95,7268],[264,7366],[740,7469],[118,7360],[299,7918],[512,7519],[373,7512],[426,7650],[222,7276],[688,7408],[454,7276],[758,7681],[731,7470],[94,7787],[0,7898],[419,7611],[611,7528],[317,7525],[670,7254],[555,7560],[487,7958],[5,7396],[260,7207],[125,7865],[92,7704],[412,7291],[472,7229],[396,7306],[211,7865],[37,7316],[106,7888],[128,7662],[218,7745],[226,7292],[680,7734],[646,7865],[89,7665],[166,7230],[250,7525],[481,7801],[389,7566],[432,7465],[12,7231],[65,7783],[65,7881],[793,7865],[753,7524],[297,7332],[799,7608],[740,7258],[726,7648],[566,7352],[297,7276],[125,7631],[286,7410],[277,7980],[705,7376],[310,7876],[724,7271],[125,7459],[484,7524],[298,7874],[405,7646],[147,7481],[770,7740],[149,7329],[234,7263],[659,7635],[300,7329],[760,7275],[81,7632],[523,7467],[409,7256],[229,7730],[611,7556],[100,7842],[718,7546],[628,7352],[243,7485],[705,7632],[400,7526],[150,7266],[771,7707],[293,7634],[263,7532],[192,7979],[239,7300],[443,7970],[276,7794],[322,7209],[440,7234],[797,7531],[661,7494],[105,7298],[707,7418]]))
if __name__ == '__main__':
    main()
