from typing import List
from tabulate import tabulate


class Solution:
    # 71 Simplify Path, Medium
    def simplifyPath(self, path: str) -> str:
        path = path.replace("//", "/")
        path = path.split("/")
        res = []
        for i, c in enumerate(path):
            if c == "" or c == ".":
                continue
            elif c == "..":
                if res != []:
                    res.pop(-1)
            else:
                res.append(c)
        return "/" + "/".join(res)

    # 72 Edit Distance, Hard
    def minDistance(self, word1: str, word2: str) -> int:
        len_1 = len(word1)
        len_2 = len(word2)
        if len_1 == 0:
            return len_2
        elif len_2 == 0:
            return len_1

        dp = [[0 for _ in range(len_1 + 1)] for _ in range(len_2 + 1)]

        for i in range(1, len_2 + 1):
            dp[i][0] = i
        for j in range(1, len_1 + 1):
            dp[0][j] = j

        for i in range(1, len_2 + 1):
            for j in range(1, len_1 + 1):
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1,
                               dp[i - 1][j - 1] + (word1[j - 1] != word2[i - 1]))

        # print(tabulate(dp))
        return dp[-1][-1]

    # 73 Set Matrix Zeroes, Medium
    def setZeroes(self, matrix: List[List[int]]) -> None:
        row = len(matrix)
        col = len(matrix[0])
        row_zero = set()
        col_zero = set()

        i = 0
        while i < row:
            j = 0
            while j < col:
                if matrix[i][j] == 0:
                    row_zero.add(i)
                    col_zero.add(j)
                j += 1
            i += 1

        row_zero = list(row_zero)
        col_zero = list(col_zero)
        # print(row_zero)
        # print(col_zero)
        for i in row_zero:
            matrix[i] = [0] * col
        for j in col_zero:
            for i in range(row):
                matrix[i][j] = 0

        # print(tabulate(matrix))

    # 74 Search a 2D Matrix, Medium
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        rows = len(matrix)
        if rows == 0:
            return False
        cols = len(matrix[0])
        if cols == 0:
            return False

        up, down = 0, rows - 1
        row = down
        while up <= down:
            mid = (down + up) // 2
            if matrix[mid][0] < target:
                up = mid + 1
            elif matrix[mid][0] > target:
                down = mid - 1
                row = down
            else:
                return True

        left, right = 0, cols - 1
        while left <= right:
            mid = (right + left) // 2
            if matrix[row][mid] < target:
                left = mid + 1
            elif matrix[row][mid] > target:
                right = mid - 1
            else:
                return True
        return False

    # 75 Sort Colors
    def sortColors(self, nums: List[int]) -> None:
        from collections import Counter
        count = Counter(nums)
        nums[:] = [0] * count[0] + [1] * count[1] + [2] * count[2]

    # 76 Minimum Window Substring, Hard
    def minWindow(self, s: str, t: str) -> str:

        # t只有一个字符的情况
        # if len(t) == 1:
        #     pos = s.find(t)
        #     if pos >= 0:
        #         return s[pos:pos+1]
        #     else:
        #         return ""
        # elif len(t) > len(s):
        #     return ""

        from collections import Counter
        count_t = Counter(t)

        start = 0
        cnt = 0
        res = ""
        min_len = float("inf")
        for i, c in enumerate(s):
            count_t[c] -= 1
            if count_t[c] >= 0:
                cnt += 1

            while cnt == len(t):
                if i - start + 1 < min_len:
                    min_len = i - start + 1
                    res = s[start: start + min_len]
                count_t[s[start]] += 1
                if count_t[s[start]] > 0:
                    cnt -= 1
                start += 1

        return res

    # 77 Combinations, Medium
    def combine(self, n: int, k: int) -> List[List[int]]:
        from itertools import combinations
        return list(combinations(range(1, n+1), k))

        # if k == 1:
        #     return [[i] for i in range(1, n + 1)]
        # from copy import deepcopy
        # def dfs(curr, num, ans):
        #     if len(curr) == k:
        #         ans.append(deepcopy(curr))
        #         return

        #     for i in range(num, n + 1):
        #         curr.append(i)
        #         dfs(curr, i + 1, ans)
        #         curr.pop()

        # ans = []
        # dfs([], 1, ans)
        # return ans

    # 78 Subsets, Medium
    def subsets(self, nums: List[int]) -> List[List[int]]:
        # use combination
        # from itertools import combinations
        # size = len(nums)
        # if nums == 1:
        #     return [[], nums]

        # res = [[]]
        # for i in range(1, size + 1):
        #     res.extend(combinations(nums, i))
        # # print(res)
        # return res

        # dfs
        def dfs(nums, index, path, res):
            res.append(list(path))

            for i in range(index, len(nums)):
                dfs(nums, i+1, path+[nums[i]], res)
        nums.sort()
        res = []
        dfs(nums, 0, [], res)
        return res

        # bfs
        # if len(nums) == 1:
        #     return [[], nums]

        # stack = [[[], 0]]
        # ans = []

        # while stack:
        #     arr, l = stack.pop()
        #     ans.append(arr)
        #     for i in range(l, len(nums)):
        #         stack.append([arr + [nums[i]], i+1])
        # return ans

    # 79 Word Search, Medium
    def exist(self, board: List[List[str]], word: str) -> bool:
        def dfs(x, y, index, visited, size_w, rows, cols):
            if index == size_w:
                return True
            if x < 0 or y < 0 or x >= rows or y >= cols or visited[x][y] or board[x][y] != word[index]:
                 return False

            visited[x][y] = True
            res = dfs(x + 1, y, index + 1, visited, size_w, rows, cols) or dfs(x, y + 1, index + 1, visited, size_w, rows, cols) or dfs(x - 1, y, index + 1, visited, size_w, rows, cols) or dfs(x, y - 1, index + 1, visited, size_w, rows, cols)

            visited[x][y] = False
            return res

        rows = len(board)
        if rows == 0:
            return False
        cols = len(board[0])
        size_w = len(word)
        if size_w == 0:
            return True
        visited = [[False for _ in range(cols)] for _ in range(rows)]

        for x in range(rows):
            for y in range(cols):
                if dfs(x, y, 0, visited, size_w, rows, cols):
                    return True

        return False

    # 80 Remove Duplicates from Sorted Array II， Medium
    def removeDuplicates(self, nums: List[int]) -> int:
        size = len(nums)
        if size <= 2:
            return
        index = 0
        i = 1
        cnt = 0 if nums[0] != nums[1] else 1
        while i < size:
            if nums[i] != nums[index]:
                index += 1
                cnt = 1
                nums[index] = nums[i]
            elif nums[i] == nums[index]:
                if cnt < 2:
                    index += 1
                    cnt += 1
                    nums[index] = nums[i]
                # index += 1
            i += 1
        del nums[index+1:]
        

def main():
    s = Solution()

    # 71
    # print(s.simplifyPath(path = "/../"))
    # 72
    # print(s.minDistance(word1 = "intention", word2 = "execution"))
    # 73
    # matrix = [[1,2,3,4],[5,0,7,8],[0,10,11,12],[13,14,15,0]]
    # print(tabulate(matrix))
    # s.setZeroes(matrix)
    # print(tabulate(matrix))
    # 74
    # print(s.searchMatrix(matrix = [[1,2,3],[10,14,15],[23,26,27]], target = 27))
    # 75
    # nums = [0,1,0]
    # print(nums)
    # s.sortColors(nums)
    # print(nums)
    # 76
    # print(s.minWindow(s = "ADOBECODEBANC", t = "ABC"))
    # 77
    # print(s.combine(2, 1))
    # 78
    # print(s.subsets([1, 2, 3]))
    # 79
    # print(s.exist(board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "SEssE"))
    # 80
    nums = []
    print(nums)
    s.removeDuplicates(nums)
    print(nums)


if __name__ == "__main__":
    main()
