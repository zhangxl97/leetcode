from collections import Counter, OrderedDict, defaultdict
from typing import List
from singly_linked_list import ListNode
from tree import TreeNode

from tabulate import tabulate

class Trie:
    def __init__(self):
        # 左子树指向表示 0 的子节点
        self.left = None
        # 右子树指向表示 1 的子节点
        self.right = None



class Solution:
    # 1779. 找到最近的有相同 X 或 Y 坐标的点, Easy
    def nearestValidPoint(self, x: int, y: int, points: List[List[int]]) -> int:
        res = -1
        
        dist = float('inf')
        for i, (px, py) in enumerate(points):
            if px == x or py == y:
                if px == x and py == y:
                    return i
                tmp = abs(px - x) + abs(py - y)
                if tmp < dist:
                    dist = tmp
                    res = i

        return res
    
    # 191. 位1的个数, Easy
    def hammingWeight(self, n: int) -> int:
        cnt = 0
        while n:
            cnt += (n % 10 == 1)
            n //= 10
        
        return cnt
    
    # 400. 第 N 位数字, Hard
    def findNthDigit(self, n: int) -> int:
        from math import ceil
        def generate():
            res = [0]
            tmp = 0
            for k in range(1,11):
                tmp += 9 * (10 ** (k - 1)) * k
                res.append(tmp)
            return res
        
        table = generate()
        
        digit = 0
        while digit < len(table) - 1 and table[digit + 1] < n:
            digit += 1

        bias = n - table[digit]

        # print(digit, table[digit], bias)

        num = 10 ** digit + ceil(bias / (digit + 1)) - 1
        # print(num)

        pos = bias % (digit + 1)
        # print(pos)

        if pos != 0:
            for _ in range(digit - pos + 1):
                num = num // 10
        return num % 10

    # 421. 数组中两个数的最大异或值, Medium
    def findMaximumXOR(self, nums: List[int]) -> int:
        # O(N^2) TLE
        # res = 0
        # for num1 in nums[:-1]:
        #     for num2 in nums[1:]:
        #         res = max(res, num1 ^ num2)
        # return res

        # 字典树解法 O(nlogC)，其中n是数组 nums 的长度，C 是数组中的元素范围，在本题中 C < 2^{31}

        # 字典树的根节点
        root = Trie()
        # 最高位的二进制位编号为 30
        HIGH_BIT = 30

        def add(num: int):
            cur = root
            for k in range(HIGH_BIT, -1, -1):
                bit = (num >> k) & 1
                if bit == 0:
                    if not cur.left:
                        cur.left = Trie()
                    cur = cur.left
                else:
                    if not cur.right:
                        cur.right = Trie()
                    cur = cur.right

        def check(num: int) -> int:
            cur = root
            x = 0
            for k in range(HIGH_BIT, -1, -1):
                bit = (num >> k) & 1
                if bit == 0:
                    # a_i 的第 k 个二进制位为 0，应当往表示 1 的子节点 right 走
                    if cur.right:
                        cur = cur.right
                        x = x * 2 + 1
                    else:
                        cur = cur.left
                        x = x * 2
                else:
                    # a_i 的第 k 个二进制位为 1，应当往表示 0 的子节点 left 走
                    if cur.left:
                        cur = cur.left
                        x = x * 2 + 1
                    else:
                        cur = cur.right
                        x = x * 2
            return x

        n = len(nums)
        x = 0
        for i in range(1, n):
            # 将 nums[i-1] 放入字典树，此时 nums[0 .. i-1] 都在字典树中
            add(nums[i - 1])
            # 将 nums[i] 看作 ai，找出最大的 x 更新答案
            x = max(x, check(nums[i]))

        return x


    # 1367. 二叉树中的列表, Medium
    def isSubPath(self, head: ListNode, root: TreeNode) -> bool:

        def check(head, root):
            if head is None:
                return True
            elif root is None:
                return False
            
            if head.val != root.val:
                return False
            
            return check(head.next, root.left) or check(head.next, root.right)

        if root is None:
            return False

        return check(head, root) or self.isSubPath(head, root.left) or self.isSubPath(head, root.right) 
    
    # 279. 完全平方数, Medium
    def numSquares(self, n: int) -> int:
        def is_square(n):
            if int(n ** 0.5) ** 2 == n:
                return True
            else:
                return False
        
        if n <= 3:
            return n

        dp = [0] * (n + 1)
        dp[0] = 1
        for i in range(1, 4):
            dp[i] = i
        
        for i in range(4, n + 1):

            if is_square(i):
                dp[i] = 1
            else:
                min_num = float('inf')
                # if i == 8:
                    # input()
                for k in range(int(i ** 0.5), 0, -1):
                    left = i - k * k
                    min_num = min(min_num, dp[k * k] + dp[left])

                dp[i] = min_num
        print(max(dp))
        return dp[n]

    # LCP 34. 二叉树染色
    # DP
    def maxValue(self, root: TreeNode, k: int) -> int:
        def dfs(root, k):
            if root is None:
                return [0] * (k + 1)
                
            left = dfs(root.left, k)
            right = dfs(root.right)

            dp = [0] * (k + 1)
            dp[0] = max(left) + max(right)
            for i in range(1, k + 1):
                tmp = 0
                for j in range(i):
                    tmp = max(tmp, left[j] + right[i - j - 1])
                dp[i] = tmp + root.val
            return dp
        
        return max(dfs(root, k))
            
    # 1498. 满足条件的子序列数目
    # 双指针 + 前缀
    def numSubseq(self, nums: List[int], target: int) -> int:
        # if min(nums) + max(nums) <= target:
        #     return 2 ** len(nums) - 1
        
        # DFS, TLE
        # size = len(nums)
        
        # self.res = 0

        # def dfs(nums, min_val, max_val, target, tmp, k):
        #     if len(tmp) == k and min_val + max_val <= target:
        #         print(tmp)
        #         self.res += 1
        #     elif nums == []:
        #         return

        #     for i in range(len(nums)):
        #         dfs(nums[i+1:], min(min_val, nums[i]), max(max_val, nums[i]), target, tmp + [nums[i]], k)

        # nums.sort()
        # for i in range(1, size + 1):
        #     dfs(nums, float('inf'), -float('inf'), target, [], i)
        # return self.res % (10 ** 9 + 7)

        # 
        # from collections import Counter

        # nums = Counter(nums)


        # keys = sorted(list(nums.keys()))
        # size = len(nums)
        # pres = [0] * size

        # pres[0] = 2 ** nums[keys[0]]
        # for i in range(1, size):
        #     pres[i] = pres[i - 1] * (2 << (nums[keys[i]] - 1))

        # left, right = 0, len(keys) - 1

        # res = 0
        # print(nums, keys)
        # print(pres)

        # while left <= right:

        #     if keys[left] + keys[right] > target:
        #         right -= 1
            
        #     else:
        #         if right == 0:
        #             res += pres[0] - 1
        #         else:
        #             # res += (2 ** nums[keys[left]] - 1) * (2 ** nums[keys[right]]) * pres[right - 1] // pres[left]
        #             if left > 0:
        #                 res += (pres[left] // pres[left - 1] - 1) * pres[right] // pres[left]
        #             else:
        #                 res += (pres[left] - 1) * pres[right] // pres[left]
        #         left += 1
                
        # return res % (10 ** 9 + 7)

        nums.sort()
        size = len(nums)
        mod = 10 ** 9 + 7
        pres = [0] * size
        pres[0] = 1
        for i in range(1, size):
            pres[i] = (pres[i - 1] << 1) % mod
        
        res = 0
        left, right = 0, size - 1
        while left <= right:
            if nums[left] + nums[right] > target:
                right -= 1
            else:
                res = (res + pres[right - left]) % mod
                left += 1
        return res

    # 1711. 大餐计数
    # 哈希表
    def countPairs(self, deliciousness: List[int]) -> int:

        def check(food1, food2):
            res = food1 + food2
            return res > 0 and res & (res - 1) == 0
        
        # O(N^2) TLE
        # res = 0
        # for i in range(len(deliciousness) - 1):
        #     for j in range(i + 1, len(deliciousness)):
        #         if check(deliciousness[i], deliciousness[j]):
        #             res += 1
        # return res
        
        # O(22N)
        def check(food1, food2):
            res = food1 + food2
            return res > 0 and res & (res - 1) == 0
        from collections import Counter
        res = 0
        deliciousness = Counter(deliciousness)
        for food1 in deliciousness.keys():
            k = 0
            while 2 ** k < food1:
                k += 1
            for i in range(k, 22):
                food2 = 2 ** i - food1
                if deliciousness[food2] > 0:
                    if food1 == food2:
                        if deliciousness[food1] > 1 and check(food1, food1):
                            res += deliciousness[food1] * (deliciousness[food1] - 1) // 2
                    else:
                        if check(food1, food2):
                            res += deliciousness[food1] * deliciousness[food2]
            deliciousness[food1] = 0


        return res % (10 ** 9 + 7)

    # 1288. 删除被覆盖区间, Medium
    def removeCoveredIntervals(self, intervals: List[List[int]]) -> int:
        intervals.sort(key=lambda x:(x[0],-x[1]))
        print(intervals)

        res = 0
        start_pre, end_pre = intervals[0][0], intervals[0][1]
        for start, end in intervals[1:]:
            if start >= start_pre and end <= end_pre:
                res += 1
            else:
                start_pre, end_pre = start, end
        return len(intervals) - res

    # 446. 等差数列划分 II, Hard
    def numberOfArithmeticSlices(self, nums: List[int]) -> int:
        size = len(nums)
        if size <= 2:
            return 0
        
        # DP
        from collections import defaultdict
        dp = [defaultdict(int) for _ in range(len(nums))] 
        res = 0
        for i in range(len(nums)):
            for j in range(i):
                diff = nums[i] - nums[j]
                dp[i][diff] += dp[j][diff] +  1
                # 说明满足长度大于等于3
                if dp[j].get(diff):
                    res += dp[j][diff]
        return res



        # DFS TLE
        # self.res = 0
        # def dfs(nums, idx, tmp):
        #     if len(tmp) > 2:
        #         print(tmp)
        #         self.res += 1
        #     if idx == len(nums):
        #         return
        #     for i in range(idx, len(nums)):
        #         if len(tmp) < 2 or (len(tmp) >= 2 and nums[i] - tmp[-1] == tmp[-1] - tmp[-2]):
        #             dfs(nums, i + 1, tmp + [nums[i]])
        #         # elif len(tmp) > 2 <= 2:
        #         #     dfs(nums, i + 1, tmp + [nums[i]])
        # dfs(nums, 0, [])
        # return self.res
    
    # 717. 1比特与2比特字符
    def isOneBitCharacter(self, bits: List[int]) -> bool:
        def back(bits):
            if bits == []:
                return True
            
            if bits[-1] == 0:
                if len(bits) > 1 and bits[-2] == 1:
                    return back(bits[:-2]) or back(bits[:-1])
                else:
                    return back(bits[:-1])
            else:
                if len(bits) > 1 and bits[-2] == 1:
                    return back(bits[:-2])
                else:
                    return False


        size = len(bits)
        if size == 1:
            return True
        elif size == 2:
            return bits[0] == 0

        
        if bits[size - 2] == 0:
            return True
        
        return not back(bits[:-2])
    # 1072. 按列翻转得到最大值等行数, Medium
    def maxEqualRowsAfterFlips(self, matrix: List[List[int]]) -> int:
        # 变换之后相等的行，在变换之前必定相等或者相反
        from collections import defaultdict
        freqs = defaultdict(int)
        for row in matrix:
            mode = ""
            if row[0] == 0:
                for c in row:
                    mode += str(c)
            else:
                for c in row:
                    mode += "1" if c == 0 else "0"
            freqs[mode] += 1
        return max(freqs.values())
    # 1020. 飞地的数量
    def numEnclaves(self, grid: List[List[int]]) -> int:
        def dfs(grid, row, col, directions):

            grid[row][col] = 0
            for dr, dc in directions:
                nr = row + dr
                nc = col + dc
                if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]) and grid[nr][nc] == 1:
                    dfs(grid, row+dr, col+dc, directions)

        rows = len(grid)
        cols = len(grid[0])
        directions = [[-1,0],[1,0],[0,1],[0,-1]]

        for col in range(cols):
            if grid[0][col] == 1:
                dfs(grid, 0, col, directions)
            if grid[rows - 1][col] == 1:
                dfs(grid, rows - 1, col, directions)
        for row in range(1, rows - 1):
            if grid[row][0] == 1:
                dfs(grid, row, 0, directions)
            if grid[row][cols - 1] == 1:
                dfs(grid, row, cols - 1, directions)
        
        cnt = 0
        for row in range(rows):
            for col in range(cols):
                if grid[row][col] == 1:
                    cnt += 1
        return cnt

    # 1551. 使数组中所有元素相等的最小操作数
    def minOperations(self, n: int) -> int:
        from math import ceil
        return int((n // 2) * (2 * ceil(n / 2) + 2 * n - 2 * (n // 2)) / 4)
    # 1754. 构造字典序最大的合并字符串
    def largestMerge(self, word1: str, word2: str) -> str:
        i1 = 0
        i2 = 0
        merge = ""
        while i1 < len(word1) or i2 < len(word2):
            if i1 == len(word1):
                merge += word2[i2:]
                break
            elif i2 == len(word2):
                merge += word1[i1:]
                break
            else:
                if word1[i1:] > word2[i2:]:
                    merge += word1[i1]
                    i1 += 1
                else:
                    merge += word2[i2]
                    i2 += 1
        return merge
    # 1318. 或运算的最小翻转次数
    def minFlips(self, a: int, b: int, c: int) -> int:
        cnt = 0
        while a != 0 or b != 0 or c != 0:
            bit_a = a % 2
            bit_b = b % 2
            bit_c = c % 2
            if bit_c == 0:
                cnt += bit_a + bit_b
            else:
                cnt += (bit_a + bit_b) == 0
            a = a >> 1
            b = b >> 1
            c = c >> 1
        return cnt

    # 1647. 字符频次唯一的最小删除次数
    def minDeletions(self, s: str) -> int:
        from collections import Counter
        s = Counter(s)
        s = Counter(list(s.values()))
        cnt = 0

        for key in range(max(s.keys()), 0, -1):
            if s[key] > 1:
                cnt += s[key] - 1
                s[key - 1] += s[key] - 1
                s[key] = 1
        return cnt
    # 1390. 四因数
    def sumFourDivisors(self, nums: List[int]) -> int:
        def four_divisors(num):
            cnt = 2
            divisors_sum = 1 + num
            for i in range(2, int(num ** 0.5 + 1)):
                if num // i * i == num:
                    if num // i == i:
                        cnt += 1
                        divisors_sum += i
                    else:
                        cnt += 2
                        divisors_sum += i + num // i
                    if cnt > 4:
                        return 0
            # print(num, cnt, divisors_sum)
            if cnt == 4:
                return divisors_sum
            else:
                return 0
    
        res = 0    
        for num in nums:
            res += four_divisors(num)
            # print(res)
        return res
    # 948. 令牌放置
    def bagOfTokensScore(self, tokens: List[int], power: int) -> int:
        from collections import deque
        tokens.sort()
        tokens = deque(tokens)
        ans = 0
        local = 0
        while tokens and (power >= tokens[0] or local > 0):
            while tokens and power >= tokens[0]:
                power -= tokens.popleft()
                local += 1
            ans = max(ans, local)
            if local > 0 and tokens:
                power += tokens.pop()
                local -= 1
        return ans

    # 1540. K 次操作转变字符串
    def canConvertString(self, s: str, t: str, k: int) -> bool:
        cnt = {}
        if len(s) != len(t):
            return False
        # for i in range(1, k + 1):
        #     if cnt.get(i % 26) is None:
        #         cnt[i % 26] = 1
        #     else:
        #         cnt[i % 26] += 1
        for i in range(len(s)):
            bais = ord(t[i]) - ord(s[i])
            if bais < 0:
                bais += 26
            # print(bais)
            if bais > 0:
                cnt.setdefault(bais, 0)
                cnt[bais] += 1
        print(cnt)
        for key in cnt:
            print(key + 26 * (cnt[key] - 1))
            if key + 26 * (cnt[key] - 1) <= k:
                continue
            else:
                return False
        return True
    # 780. 到达终点
    def reachingPoints(self, sx: int, sy: int, tx: int, ty: int) -> bool:

        # BFS TLE
        # queue = [[sx, sy]]

        # while queue != []:
        #     size = len(queue)

        #     for _ in range(size):
        #         x, y = queue.pop(0)
        #         if x == tx and y == ty:
        #             return True
        #         nx, ny = x, x+y
        #         if 1 <= nx <= tx and 1 <= ny <= ty:
        #             queue.append([nx, ny])
        #         nx, ny = x+y, x
        #         if 1 <= nx <= tx and 1 <= ny <= ty:
        #             queue.append([nx, ny])
        # return False

        # backtrack
        # print(sx, sy, tx, ty)
        if sx > tx or sy > ty:
            return False
        elif sx == tx and sy == ty:
            return True
        
        elif tx > ty:
            num = (tx - sx) // ty * ty
            return self.reachingPoints(sx, sy, tx - max(num, ty), ty)
        elif tx < ty:
            num = (ty - sy) // tx * tx
            return self.reachingPoints(sx, sy, tx, ty - max(num, tx))
        else:
            return self.reachingPoints(sx, sy, tx - ty, ty) or self.reachingPoints(sx, sy, tx, ty - tx)

# 460. LFU 缓存, Hard
class LFUCache:
    def __init__(self, capacity: int):
        self.elements = OrderedDict()
        self.capacity = capacity
        self.key2cnt = Counter()
        self.cnt2key = OrderedDict()

    def get(self, key: int) -> int:
        if self.elements.get(key) is None or self.capacity == 0:
            return -1
        else:
            self.key2cnt[key] += 1
            cnt = self.key2cnt[key]
            if cnt > 1:
                self.cnt2key[cnt - 1].remove(key)
            self.cnt2key.setdefault(cnt, [])
            self.cnt2key[cnt].append(key)
            print(self.cnt2key)
            print(self.elements)
            return self.elements[key]

    def put(self, key: int, value: int) -> None:
        if self.capacity == 0:
            return 

        if self.elements.get(key) is not None or len(self.elements) < self.capacity:  # 存在这个key，则本次操作不会导致溢出; 或者还没存满
            self.elements[key] = value
            self.key2cnt[key] += 1
            cnt = self.key2cnt[key]
            if cnt > 1:
                self.cnt2key[cnt - 1].remove(key)
            self.cnt2key.setdefault(cnt, [])
            self.cnt2key[cnt].append(key)

        else: # 不存在且溢出
            key_cnt = 1
            while self.cnt2key[key_cnt] == []:
                key_cnt += 1
            key_del = self.cnt2key[key_cnt].pop(0)
            self.key2cnt.pop(key_del)
            self.elements.pop(key_del)

            self.elements[key] = value
            self.cnt2key[1].append(key)
            self.key2cnt[key] = 1

        print(self.cnt2key)
        print(self.elements)

        


def main():
    s = Solution()

    # print(s.nearestValidPoint(x = 3, y = 4, points = [[2,3]]))
    # print(s.hammingWeight(1011))

    # print(s.findMaximumXOR(nums = [3,10,5,25,2,8]))
    # print(s.findNthDigit(10))
    # print(s.numSquares(100000))

    # print(s.numSubseq(nums = [5,2,4,1,7,6,8], target = 16))
    # print(s.countPairs(deliciousness = [1048576,1048576]))

    # print(s.removeCoveredIntervals(intervals = [[1,4],[1,2]]))
    # lFUCache = LFUCache(1)
    # lFUCache.put(0, 0)
    # print(lFUCache.get(0))
    # lFUCache.put(1, 1)
    # lFUCache.put(2, 2)
    # print(lFUCache.get(1))
    # lFUCache.put(3, 3)
    # print(lFUCache.get(2))
    # print(lFUCache.get(3))
    # lFUCache.put(4, 4)
    # print(lFUCache.get(1))
    # print(lFUCache.get(3))
    # print(lFUCache.get(4))
    # print(s.numberOfArithmeticSlices([7,7,7,7,7]))

    # print(s.isOneBitCharacter([1,0,0]))
    # print(s.maxEqualRowsAfterFlips([[0,1],[1,1]]))
    # print(s.numEnclaves([[0,0,0,0],[1,0,1,0],[0,1,1,0],[0,0,0,0]]))

    # print(s.minOperations(2))
    # print(s.largestMerge(word1 = "abcabc", word2 = "abdcaba"))
    # print(s.minFlips(a = 1, b = 2, c = 3))

    # print(s.minDeletions("bbcebab"))
    # print(s.sumFourDivisors([21,4,7,24]))
    # print(s.bagOfTokensScore(tokens = [100,200,300,400], power = 200))

    # print(s.canConvertString(s = "aab", t = "bbb", k = 27))
    print(s.reachingPoints(35,
13,
455955547,
420098884))

if __name__ == "__main__":
    main()
