from typing import List
from tabulate import tabulate

class Solution:
    # 31. Next Permutation
    def nextPermutation(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        left_i = len(nums) - 2
        right_i = len(nums) - 1

        # e.g. 2 3 1
        # left_i: 1, right_i: 2
        while left_i >= 0 and nums[left_i] >= nums[left_i + 1]:
            left_i -= 1
        # left_i: 0
        if left_i >= 0:
            # 2 3 1
            while nums[right_i] <= nums[left_i]:
                right_i -= 1
            nums[left_i], nums[right_i] = nums[right_i], nums[left_i]
        nums[left_i + 1:] = sorted(nums[left_i + 1:])

    # 32. Longest Valid Parentheses
    def longestValidParentheses(self, s: str) -> int:
        # use stack
        # max_len = 0
        # stack = []
        # last = -1
        # for i in range(len(s)):
        #     if s[i] == '(':
        #         stack.append(i)  # push the INDEX into the stack!!!!
        #     else:
        #         if stack == []:
        #             last = i
        #         else:
        #             stack.pop()
        #             if stack == []:
        #                 max_len = max(max_len, i - last)
        #             else:
        #                 max_len = max(max_len, i - stack[-1])
        # return max_len

        # use dp
        size = len(s)
        if size < 2:
            return 0
        dp = [0 for _ in range(size)]
        for i in range(1, size):
            if s[i] == ')':
                j = i - 1 - dp[i - 1]  # 直接去查找前面的第j位移过了dp[i-1]位已经匹配的
                if j >= 0 and s[j] == '(':  # 如果那位是‘（’则可以总数多+2
                    dp[i] = dp[i - 1] + 2
                    if j - 1 >= 0:
                        dp[i] += dp[j - 1]  # 重点，会把这次匹配之前的加进去，例如（）（（））
        return max(dp)

    # 33. Search in Rotated Sorted Array
    def search(self, nums: List[int], target: int) -> int:
        # left, right = 0, size - 1
        # while left <= right:
        #     mid = (left + right) // 2
        #     if target == nums[mid]:
        #         return True
        #     # nums[left to mid] is sorted 
        #     if nums[left] <= nums[mid]:
        #         if target < nums[mid] and target >= nums[left]:
        #             right = mid - 1
        #         else:
        #             left = mid + 1
        #     # nums[mid to right] is sorted
        #     else:
        #         if target > nums[mid] and target <= nums[right]:
        #             left = mid + 1
        #         else:
        #             right = mid - 1

        # return False
        size = len(nums)
        if size == 1:
            if nums[0] == target:
                return 0
            else:
                return -1

        index = 0
        while index < size:
            if target == nums[index]:
                return index
            elif target < nums[index]:
                while index < size - 1 and nums[index] < nums[index + 1]:
                    index += 1
                index += 1
            else:
                index += 1
        return -1

    # 34. Find First and Last Position of Element in Sorted Array
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        size = len(nums)
        if size == 0:
            return [-1, -1]
        elif size == 1:
            if nums[0] == target:
                return [0, 0]
            else:
                return [-1, -1]

        left = 0
        right = size - 1
        while left <= right:
            mid = (left + right) >> 1
            if nums[mid] == target:
                mid_left = mid
                mid_right = mid
                while mid_left > 0 and nums[mid_left - 1] == target:
                    mid_left -= 1
                while mid_right < size - 1 and nums[mid_right + 1] == target:
                    mid_right += 1
                return [mid_left, mid_right]
            elif nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return [-1, -1]

    # 35. Search Insert Position
    def searchInsert(self, nums: List[int], target: int) -> int:
        from bisect import bisect_left
        return bisect_left(nums, target)

    # 36. Valid Sudoku
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        # from collections import Counter
        # import numpy as np
        # board = np.array(board)
        # # validate cols
        # for col in range(9):
        #     temp = Counter(board[:, col])
        #     for index in temp.keys():
        #         if index != '.' :
        #             if temp.get(str(index)) is None or temp.get(str(index)) == 1:
        #                 continue
        #             else:
        #                 return False
        # # validate rows
        # for row in range(9):
        #     temp = Counter(board[row, :])
        #     for index in temp.keys():
        #         if index != '.' :
        #             if temp.get(str(index)) is None or temp.get(str(index)) == 1:
        #                 continue
        #             else:
        #                 return False
        # # validate 3*3 blocks
        # for row in range(0,9,3):
        #     for col in range(0,9,3):
        #         temp = Counter(board[row:row+3,col:col+3].reshape(9,))
        #         for index in temp.keys():
        #             if index != '.':
        #                 if temp.get(str(index)) is None or temp.get(str(index)) == 1:
        #                     continue
        #                 else:
        #                     return False
        #
        # return True

        # save all positions of each number
        # numbers = []
        # for i, row in enumerate(board):
        #     for j, c in enumerate(row):
        #         if c != '.':
        #             numbers += [(i, c), (c, j), (i // 3, j // 3, c)]
        #
        # return len(set(numbers)) == len(numbers)
        for x in range(9):
            for y in range(9):
                if board[x][y] != '.':
                    tmp = board[x][y]
                    board[x][y] = '.'
                    for row in range(9):
                        if board[row][y] == tmp:
                            return False
                    for col in range(9):
                        if board[x][col] == tmp:
                            return False
                    for row in range(3):
                        for col in range(3):
                            if board[(x // 3) * 3 + row][(y // 3) * 3 + col] == tmp:
                                return False
                    board[x][y] = tmp
        return True

    # 37. Sudoku Solver
    def dfs(self, board: List[List[str]]):
        def is_valid(x,y):
            tmp = board[x][y]
            board[x][y] = '.'
            for row in range(9):
                if board[row][y] == tmp:
                    return False
            for col in range(9):
                if board[x][col] == tmp:
                    return False
            for row in range(3):
                for col in range(3):
                    if board[(x // 3) * 3 + row][(y // 3) * 3 + col] == tmp:
                        return False
            board[x][y] = tmp
            return True


        for col in range(9):
            for row in range(9):
                if board[row][col] == '.':
                    for c in "123456789":
                        board[row][col] = c
                        if is_valid(row,col) and self.dfs(board):
                            return True
                        board[row][col] = '.'
                    return False
        return True

    def solveSudoku(self, board: List[List[str]]) -> None:
        self.dfs(board)

    # 38. Count and Say
    def countAndSay(self, n: int) -> str:
        if n == 1:
            return "1"
        elif n == 2:
            return "11"
        curr = "11"
        res = ""

        for i in range(n - 2):
            res = ""
            size = len(curr)
            fre = 1
            for i in range(1, size):
                if curr[i - 1] == curr[i]:
                    fre += 1
                    continue
                else:
                    res += str(fre) + curr[i - 1]
                    fre = 1
            res += str(fre) + curr[-1]
            curr = res
        return res

    # 39. Combination Sum
    # def combination_search(self, candidates, target, res, curr):
    #     if target == 0:
    #         if sorted(curr) not in res:
    #             res.append(sorted(curr))
    #         return
    #
    #     for i, num in enumerate(candidates):
    #         curr.append(num)
    #         if target - num >= candidates[0] or target - num == 0:
    #             self.combination_search(candidates, target - num, res, curr)
    #         curr.pop()
    # def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
    #     candidates.sort()
    #     res = []
    #     curr = []
    #     self.combination_search(candidates, target, res, curr)
    #     return res


    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:

        res = []

        def dfs(set, target, idx):
            if target == 0:
                res.append(set)
                return
            for i in range(idx, len(candidates)):
                if candidates[i] <= target:
                    dfs(set + [candidates[i]], target - candidates[i], i)

        dfs([], target, 0)

        return res

    # 40 Combination Sum
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:

        res = []
        candidates.sort()
        def dfs(set, target, idx):
            if target == 0:
                res.append(set)
                return
            i = idx
            size = len(candidates)
            while i < size:
                if candidates[i] <= target:
                    dfs(set + [candidates[i]], target - candidates[i], i + 1)
                while i < size - 1 and candidates[i] == candidates[i + 1]:
                    i += 1
                i += 1

        dfs([], target, 0)

        return res




def main():
    s = Solution()

    # 31
    # nums = [2, 3, 1]
    # print(nums)
    # s.nextPermutation(nums)
    # print(nums)
    # 32
    # print(s.longestValidParentheses("(()())"))
    # 33
    # print(s.search(nums = [1], target = 0))
    # 34
    # print(s.searchRange(nums = [5,7,7,8,8,10], target = 6))
    # 35
    # print(s.searchInsert([1,3,5,6], 0))
    # 36
    # print(s.isValidSudoku([[".", ".", ".", ".", ".", "3", ".", ".", "."],
    #                        ["8", ".", ".", ".", ".", "5", ".", "1", "."],
    #                        [".", ".", ".", ".", "7", ".", ".", ".", "3"],
    #                        [".", ".", ".", ".", ".", ".", ".", ".", "."],
    #                        [".", "5", "9", "7", ".", ".", ".", "9", "."],
    #                        ["7", ".", ".", ".", ".", ".", ".", ".", "."],
    #                        [".", "6", ".", ".", ".", ".", "2", ".", "."],
    #                        [".", ".", ".", ".", ".", ".", ".", ".", "."],
    #                        [".", ".", ".", ".", ".", ".", "7", ".", "."]]))
    # board = [[".",".",".","2",".",".",".","6","3"],
    #          ["3",".",".",".",".","5","4",".","1"],
    #          [".",".","1",".",".","3","9","8","."],
    #          [".",".",".",".",".",".",".","9","."],
    #          [".",".",".","5","3","8",".",".","."],
    #          [".","3",".",".",".",".",".",".","."],
    #          [".","2","6","3",".",".","5",".","."],
    #          ["5",".","3","7",".",".",".",".","8"],
    #          ["4","7",".",".",".","1",".",".","."]]
    #
    # print(tabulate(board))
    # s.solveSudoku(board)
    # print(tabulate(board))
    # 38
    # print(s.countAndSay(6))
    # 39
    print(s.combinationSum(candidates = [2,3,6,7], target = 7,))
    # 40
    # print(s.combinationSum2(candidates = [2,5,2,1,2], target = 5,))

if __name__ == '__main__':
    main()
