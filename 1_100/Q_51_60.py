from typing import List
from tabulate import tabulate 
from time import time

class Solution:
    # N-Queens
    def solveNQueens(self, n: int) -> List[List[str]]:
        if n == 1:
            return [["Q"]]
        elif n == 2:
            return []
        
        # Time Limited
        # from copy import deepcopy
        # def check(tmp, n):
        #     flag_vertical = {i:0 for i in range(n)}
        #     flag_horizontal = {i:0 for i in range(n)}
        #     flag_leftup2rightdown = {}
        #     flag_rightup2leftdown = {}

        #     for i in range(n):
        #         for j in range(n):
        #             if tmp[i][j] == "Q":
        #                 flag_vertical[j] += 1
        #                 if flag_vertical[j] > 1:
        #                     return False
        #                 flag_horizontal[i] += 1
        #                 if flag_horizontal[i] > 1:
        #                     return False
        #                 if flag_leftup2rightdown.get(i - j) is None:
        #                     flag_leftup2rightdown[i - j] = 1
        #                 else:
        #                     flag_leftup2rightdown[i - j] += 1
        #                 if flag_leftup2rightdown[i - j] > 1:
        #                     return False
        #                 if flag_rightup2leftdown.get(i + j) is None:
        #                     flag_rightup2leftdown[i + j] = 1
        #                 else:
        #                     flag_rightup2leftdown[i + j] += 1
        #                 if flag_rightup2leftdown[i + j] > 1:
        #                     return False

        #     for v in flag_vertical.values():
        #         if v != 1:
        #             return False
        #     for v in flag_horizontal.values():
        #         if v != 1:
        #             return False
        #     for v in flag_leftup2rightdown.values():
        #         if v != 1:
        #             return False
        #     for v in flag_rightup2leftdown.values():
        #         if v != 1:
        #             return False
        #     return True

        # def search(tmp, start_row, res):
        #     if start_row == n and check(tmp, n):
        #         res.append(deepcopy(tmp))
        #         return 

        #     for i in range(start_row, n):
        #         for j in range(n):
        #             tmp[i][j] = "Q"
        #             search(tmp, start_row+1, res)
        #             tmp[i][j] = "."



        # res = []
        # tmp = [["." for _ in range(n)] for _ in range(n)]
        # search(tmp, 0, res)
        # for result in res:
        #     for i in range(n):
        #         result[i] = "".join(result[i])
        # return res
        from collections import defaultdict
        board = [['.' for j in range(n)] for i in range(n)]
        rows = defaultdict(bool)
        cols = defaultdict(bool)
        diag1 = defaultdict(bool)  # rightup2leftdown
        diag2 = defaultdict(bool)  # leftup2rightdown

        def available(x, y):
            return not rows[x] and not cols[y] and not diag1[x+y] and not diag2[x-y]
        
        def update(x, y, flag):
            rows[x] = flag
            cols[y] = flag
            diag1[x+y] = flag
            diag2[x-y] = flag
            board[x][y] = 'Q' if flag==True else '.'
        
        def dfs(x):
            if x == n:
                res.append([''.join(lst) for lst in board])
                return
            for y in range(n):
                if available(x , y):
                    update(x, y, True)
                    dfs(x+1)
                    update(x, y, False)       
                    
        res = []
        dfs(0)
        return res
    
    # 52. N-Queens II
    def totalNQueens(self, n: int) -> int:
        from collections import defaultdict
        board = [['.' for j in range(n)] for i in range(n)]
        rows = defaultdict(bool)
        cols = defaultdict(bool)
        diag1 = defaultdict(bool)  # rightup2leftdown
        diag2 = defaultdict(bool)  # leftup2rightdown

        def available(x, y):
            return not rows[x] and not cols[y] and not diag1[x+y] and not diag2[x-y]
        
        def update(x, y, flag):
            rows[x] = flag
            cols[y] = flag
            diag1[x+y] = flag
            diag2[x-y] = flag
            board[x][y] = 'Q' if flag==True else '.'
        
        def dfs(x):
            if x == n:
                res.append(True)
                return
            for y in range(n):
                if available(x , y):
                    update(x, y, True)
                    dfs(x+1)
                    update(x, y, False)       
                    
        res = []
        dfs(0)
        return len(res) 

    # 53 Maximum Subarray, Easy
    def maxSubArray(self, nums: List[int]) -> int:
        # dp = [0]*len(nums)
        # for i,num in enumerate(nums):            
        #     dp[i] = max(dp[i-1] + num, num)
        # return max(dp)
        max_sum_until_i = max_sum= nums[0]
        for num in nums[1:]:
            max_sum_until_i = max(max_sum_until_i+num, num)
            max_sum = max(max_sum, max_sum_until_i)
        return max_sum
    
    # 54 Spiral Matrix， Medium
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        # # very slow
        # if matrix == [] or matrix == [[]]:
        #     return []
        # elif len(matrix) == 1:
        #     return matrix[0]
        # import numpy as np
        # matrix = np.array(matrix)
        # row, col = matrix.shape
        # up, down, left, right = 0, row - 1, 0, col - 1
        # res = []
        # while up <= down and left <= right:
        #     res.extend(matrix[up, left : right + 1])
        #     res.extend(matrix[up + 1 : down + 1, right])
        #     if down > up:
        #         res.extend(matrix[down, left : right][::-1])
        #     if right > left:
        #         res.extend(matrix[up + 1 : down, left][::-1])
        #     up += 1
        #     down -= 1
        #     left += 1
        #     right -= 1
        # return res
        def rotate(m):
            # zip(*m)将其转换为按列对应的迭代器
            # map()根据提供的函数对指定序列做映射，python3返回迭代器
            m = list(map(list, zip(*m)))  # ==> 转置操作！！！
            m.reverse()   # 上下颠倒，类似np.flipud()
            return m
    
        res = []
        while matrix:
            res += matrix[0]
            matrix = rotate(matrix[1:])
        return res

    # 55 Jump Game, Medium
    def canJump(self, nums: List[int]) -> bool:
   
        curr = 0
        size = len(nums)
        for i in range(size):
            if nums[i] == 0:
                if curr > i:
                    continue
                else:
                    break
            curr = max(curr, i+nums[i])
            
    
        return curr >= size - 1
    
    # 56 Merge Intervals, Medium
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals.sort()
        size = len(intervals)
        if size <= 1:
            return intervals
        ans = []
        start_pre, end_pre = intervals[0]
        for i in range(1, size):
            start, end = intervals[i]
            if start <= end_pre:
                end_pre = max(end, end_pre)
            else:
                ans.append([start_pre, end_pre])
                start_pre = start
                end_pre = end
        ans.append([start_pre, end_pre])
        return ans

    # 57 Insert Interval, Medium
    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        intervals.append(newInterval)
        intervals.sort()
        size = len(intervals)
        if size <= 1:
            return intervals
        ans = []
        start_pre, end_pre = intervals[0]
        for i in range(1, size):
            start, end = intervals[i]
            if start <= end_pre:
                end_pre = max(end, end_pre)
            else:
                ans.append([start_pre, end_pre])
                start_pre = start
                end_pre = end
        ans.append([start_pre, end_pre])
        return ans

    # 58 Length of Last Word, Easy
    def lengthOfLastWord(self, s: str) -> int:
        s = s.strip().split(' ')
        if s[-1]:
            return len(s[-1])
        else:
            return 0

    # 59 Spiral Matrix II
    def generateMatrix(self, n: int) -> List[List[int]]:
        tmp = [[j + i * n + 1 for j in range(n)] for i in range(n)]
        res = tmp        
        rotate_order = []
        while tmp:
            rotate_order += tmp[0]
            tmp = tmp[1:]
            tmp = list(map(list, zip(*tmp)))
            tmp.reverse()

        for i, num in enumerate(rotate_order):
            num -= 1
            res[num//n][num%n] = i + 1

        return res

    # 60 Permutation Sequence
    def getPermutation(self, n: int, k: int) -> str:
        # from itertools import permutations
        # per = permutations(range(1, n + 1))
        # last = None
        # for i in range(k):
        #     last = next(per)
        # return ''.join(str(c) for c in last)
        import math
        res = ''
        digits = [str(i + 1) for i in range(n)]
        t = k - 1
        for i in range(n, 0, -1):
            ind = t//math.factorial(i - 1)
            t%=math.factorial(i - 1)
            if t == 0:
                res += digits[ind] + "".join(digits[:ind] + digits[ind + 1:])
                return res
            else:
                res += digits[ind]
                del digits[ind]
        return res


def main():
    s = Solution()

    # 51
    # tic = time()
    # print(s.solveNQueens(5))
    # print(time() - tic)
    # 52
    # print(s.totalNQueens(4))       
    # 53
    # print(s.maxSubArray([-2,1,-3,4,-1,2,1,-5,4]))
    # 54
    # print(s.spiralOrder(
    #                     [
    #                     [ 1, 2, 3 ],
    #                     [ 4, 5, 6 ],
    #                     [ 7, 8, 9 ]
    #                     ]))
    # 55
    # print(s.canJump([3,0,8,2,0,0,1]))
    # 56
    # print(s.merge([[5,42],[4,6]]))
    # 57
    # print(s.insert(intervals = [[1,5]], newInterval = [2,7]))
    # 58
    # print(s.lengthOfLastWord("  "))
    # 59
    # print(s.generateMatrix(5))
    # 60
    print(s.getPermutation(4,9))

if __name__ == "__main__":
    main()
