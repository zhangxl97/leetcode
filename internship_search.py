from typing import List
from tabulate import tabulate
from tree import TreeNode

class Solution:
    # BFS，可以求解最短路径等   **最优解**   问题
    # 1091. 二进制矩阵中的最短路径, Medium
    def shortestPathBinaryMatrix(self, grid: List[List[int]]) -> int:
        if grid[0][0] == 1:
            return -1
        
        rows = len(grid)
        cols = len(grid[0])
        # 上，下，左，右，左上，左下，右上，右下
        directions = [[-1,0],[1,0],[0,-1],[0,1],[-1,-1],[1,-1],[-1,1],[1,1]]

        queue = [[0,0]]
        length = 0
        while queue != []:
            size = len(queue)
            length += 1
            # print(queue)
            while size > 0:
                size -= 1
                row, col = queue.pop(0)
                if grid[row][col] == 1:
                    continue
                if row == rows - 1 and col == cols - 1:
                    return length
                
                grid[row][col] = 1

                for direction in directions:
                    # print(direction)
                    d_row, d_col = direction[0], direction[1]
                    new_row = row + d_row
                    new_col = col + d_col
                    if new_row < 0 or new_col < 0 or new_row >= rows or new_col >= cols or grid[new_row][new_col] == 1:
                        continue
                    queue.append([new_row, new_col])
        return -1

    # 279. 完全平方数, Medium
    # 也可由dp求解
    # 用BFS TLE
    def numSquares(self, n: int) -> int:
        def generate_squares(n):
            square = 1
            diff = 3
            ans = []
            while square <= n:
                ans.append(square)
                square += diff
                diff += 2
            return ans

        squares = generate_squares(n)
        visited = [False] * (n + 1)
        visited[n] = True
        queue = [n]
        cnt = -1
        while len(queue) > 0:
            cnt += 1
            size = len(queue)
            while size > 0:
                size -= 1
                num = queue.pop(0)

                visited[num] = True
                if num == 0:
                    return cnt
                
                for square in squares:
                    next_num = num - square
                    if next_num < 0 or visited[next_num]:
                        continue
                    queue.append(next_num)

    # 127. 单词接龙, Hard
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:

        def connection_between_words(word1, word2):
            diff = 0
            for i in range(len(word1)):
                if word1[i] != word2[i]:
                    diff += 1
                    if diff > 1:
                        return False
            return diff == 1

        def generate_graphic(wordList):
            size = len(wordList)
            graph = {word:[] for word in wordList}
            for word1 in wordList:
                for word2 in wordList:
                    if word1 != word2 and connection_between_words(word1, word2):
                        graph[word1].append(word2)
                        graph[word2].append(word1)
            return graph

        if endWord not in wordList:
            return 0
        
        if beginWord in wordList:
            wordList.remove(beginWord)
        wordList = list(set(wordList))
        wordList = [beginWord] + wordList
        
        graph = generate_graphic(wordList)
        # print(tabulate(connection))

        visited = {word:False for word in wordList}
        
        cnt = 0
        queue = [beginWord]

        while queue != []:
            # print(queue)
            size = len(queue)
            cnt += 1
            while size > 0:
                size -= 1
                word = queue.pop(0)

                if visited[word]:
                    continue

                if word == endWord:
                    return cnt
                visited[word] = True

                queue.extend(graph[word])
            queue = list(set(queue))

        return 0

    # DFS 求解这种   **可达性**   问题
    # 695. 岛屿的最大面积, Medium
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        directions = [[-1,0],[1,0],[0,-1],[0,1]]
        def dfs(grid, r, c, rows, cols, directions):
            if r < 0 or c < 0 or r == rows or c == cols or grid[r][c] == 0:
                return 0
            
            area = 1
            grid[r][c] = 0

            for direction in directions:
                d_row, d_col = direction[0], direction[1]
                nr = r + d_row
                nc = c + d_col
                area += dfs(grid, nr, nc, rows, cols, directions)

            return area

        
        rows = len(grid)
        cols = len(grid[0])

        max_area = 0
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 1:
                    max_area = max(max_area, dfs(grid, r, c, rows, cols, directions))

        return max_area

    # 200. 岛屿数量, Medium
    # grid为该位置是否为物体
    def numIslands(self, grid: List[List[str]]) -> int:
        def dfs(grid, row, col, rows, cols, directions):
            if row < 0 or col < 0 or row == rows or col == cols or grid[row][col] == "0":
                return 
            
            grid[row][col] = "0"

            for d_row, d_col in directions:
                nr = row + d_row
                nc = col + d_col
                dfs(grid, nr, nc, rows, cols, directions)
        
        rows = len(grid)
        cols = len(grid[0])
        directions = [[1,0],[-1,0],[0,-1],[0,1]]
        cnt = 0
        for row in range(rows):
            for col in range(cols):
                if grid[row][col] == "1":
                    dfs(grid, row, col, rows, cols, directions)
                    cnt += 1
        return cnt

    # 547. 省份数量, Medium
    # isConnected为i，j的连接关系
    def findCircleNum(self, isConnected: List[List[int]]) -> int:

        def dfs(isConnected, visited, city, city_num):
            visited[city] = True
            for i in range(city_num):
                if i != city and visited[i] is False and isConnected[city][i] == 1:
                    dfs(isConnected, visited, i, city_num)


        city_num = len(isConnected)
        # cols = len(isConnected[0])

        visited = [False] * city_num

        province = 0
        for i in range(city_num):
            if visited[i] is False:
                dfs(isConnected, visited, i, city_num)
                province += 1
        return province

    # 130. 被围绕的区域, Medium
    def solve(self, board: List[List[str]]) -> None:

        def dfs(board, row, col, rows, cols, directions):
            if row < 0 or col < 0 or row == rows or col == cols or board[row][col] != "O":
                return 
            
            board[row][col] = "D"

            for d_row, d_col in directions:
                nr = row + d_row
                nc = col + d_col
                dfs(board, nr, nc, rows, cols, directions)
        
        rows = len(board)
        cols = len(board[0])
        directions = [[1,0],[-1,0],[0,1],[0,-1]]
        for i in range(rows):
            dfs(board, i, 0, rows, cols, directions)  # 最左
            dfs(board, i, cols - 1, rows, cols, directions)  # 最右

        for j in range(1, cols - 1):
            dfs(board, 0, j, rows, cols, directions)  # 最上
            dfs(board, rows - 1, j, rows, cols, directions)  # 最下
    
        for i in range(rows):
            for j in range(cols):
                if board[i][j] == "O":
                    board[i][j] = "X"
                elif board[i][j] == "D":
                    board[i][j] = "O"

    # 417. 太平洋大西洋水流问题, Medium
    def pacificAtlantic(self, heights: List[List[int]]) -> List[List[int]]:

        def dfs(heights, can_reach, row, col, rows, cols, directions):
            if can_reach[row][col]:
                return 

            can_reach[row][col] = True
            for d_row, d_col in directions:
                nr = row + d_row
                nc = col + d_col
                if nr < 0 or nc < 0 or nr == rows or nc == cols or heights[nr][nc] < heights[row][col]:
                    continue

                dfs(heights, can_reach, nr, nc, rows, cols, directions)

        
        rows = len(heights)
        cols = len(heights[0])
        can_reach_p = [[False for _ in range(cols)] for col in range(rows)]
        can_reach_A = [[False for _ in range(cols)] for col in range(rows)]
        directions = [[1,0],[-1,0],[0,1],[0,-1]]

        for col in range(cols):
            dfs(heights, can_reach_p, 0, col, rows, cols, directions)  # 最上 --> pacific
            dfs(heights, can_reach_A, rows - 1, col, rows, cols, directions)  # 最下 --> atlantic

        for row in range(rows):
            dfs(heights, can_reach_p, row, 0, rows, cols, directions)  # 最左 --> pacific
            dfs(heights, can_reach_A, row, cols - 1, rows, cols, directions)  # 最右 --> atlantic
        
        ans = []
        for row in range(rows):
            for col in range(cols):
                if can_reach_A[row][col] and can_reach_p[row][col]:
                    ans.append([row, col])
        return ans


    # backtracking 求解   **排列组合**   问题
    # 17. 电话号码的字母组合, Medium
    def letterCombinations(self, digits: str) -> List[str]:

        kvs = {"2": "abc", "3":"def", "4":"ghi", "5":"jkl", "6":"mno", "7":"pqrs", "8":"tuv", "9":"wxyz"}

        def dfs(digits, chars):
            if digits == "":
                return
            
            size = len(chars)
            while size > 0:
                c = chars.pop(0)
                size -= 1
                for d in kvs[digits[0]]:
                    chars.append(c + d)
            dfs(digits[1:], chars)

        if digits == "":
            return []
        chars = list(kvs[digits[0]])
        dfs(digits[1:], chars)
        return chars
                
    # 93. 复原 IP 地址, Medium
    def restoreIpAddresses(self, s: str) -> List[str]:

        def dfs(s, tmp, results):
            if s == "":
                if len(tmp) == 4:
                    results.append('.'.join(tmp))
                return 
            elif len(tmp) == 4:
                return 
            

            dfs(s[1:], tmp + [s[0]], results)
            if s[0] != '0':
                if len(s) >= 3 and int(s[:3]) <= 255:
                    dfs(s[3:], tmp + [s[:3]], results)
                if len(s) >= 2:
                    dfs(s[2:], tmp + [s[:2]], results)

        results = []
        dfs(s, [], results)
        return results

    # 79. 单词搜索, Medium
    def exist(self, board: List[List[str]], word: str) -> bool:
        def dfs(board, visited, index, word, row, col, rows, cols, directions):
            if index == len(word) - 1:
                return True

            visited[row][col] = True
            for d_row, d_col in directions:
                nr = row + d_row
                nc = col + d_col
                if nr < 0 or nc < 0 or nr == rows or nc == cols or board[nr][nc] != word[index+1] or visited[nr][nc] is True:
                    continue
                else:
                    flag = dfs(board, visited, index+1, word, nr, nc, rows, cols, directions)
                    if flag: 
                        return True
            visited[row][col] = False

            return False


        rows = len(board)
        cols = len(board[0])
        directions = [[0,1],[0,-1],[1,0],[-1,0]]
        visited = [[False for _ in range(cols)] for _ in range(rows)]
        # visited = None
        flag = False
        for row in range(rows):
            for col in range(cols):
                if board[row][col] == word[0]:
                    if dfs(board, visited, 0, word, row, col, rows, cols, directions):
                        return True
        return False

    # 257. 二叉树的所有路径, Easy
    def binaryTreePaths(self, root: TreeNode) -> List[str]:
        def dfs(root, tmp, results):
            if root.left is None and root.right is None:
                tmp += [str(root.val)]
                results.append('->'.join(tmp))
                return 
            
            if root.left:
                dfs(root.left, tmp + [str(root.val)], results)
            if root.right:
                dfs(root.right, tmp + [str(root.val)], results)
        
        results = []
        dfs(root, [], results)
        return results
    
    # 46. 全排列, Medium
    def permute(self, nums: List[int]) -> List[List[int]]:
        def dfs(nums, tmp, results):
            if nums == []:
                results.append(tmp)
                return 
            
            for i, num in enumerate(nums):
                dfs(nums[:i] + nums[i+1:], tmp + [num], results)

        results = []
        dfs(nums, [], results)
        return results

    # 47. 全排列 II, Medium
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        def dfs(nums, tmp, results):
            if nums == []:
                results.append(tmp)
                return 
            
            for i, num in enumerate(nums):
                if i > 0 and nums[i] == nums[i - 1]:
                    continue
                dfs(nums[:i] + nums[i + 1: ], tmp + [num], results)
        
        nums.sort()
        results = []
        dfs(nums, [], results)
        return results

    # 77. 组合, Medium
    def combine(self, n: int, k: int) -> List[List[int]]:
        def dfs(nums, tmp, results, k):
            if len(tmp) == k:
                results.append(tmp)
            
            for i, num in enumerate(nums):
                dfs(nums[i+1:], tmp+[num], results, k)
        
        nums = [i+1 for i in range(n)]
        results = []
        dfs(nums, [], results, k)
        return results

    # 39. 组合总和, Medium
    def combinationSum(self, candidates: List[int], target: int) :
        def dfs(candidates, left, tmp, results, idx):
            if left == 0:
                results.append(tmp)
                return 
            
            for i in range(idx, len(candidates)):
                candidate = candidates[i]
                if candidate <= left:
                    dfs(candidates, left - candidate, tmp + [candidate], results, i)
            
        candidates.sort()
        results = []
        dfs(candidates, target, [], results, 0)
        return results

    # 216. 组合总和 III, Medium
    def combinationSum3(self, k: int, n: int) -> List[List[int]]:

        def dfs(k, target, tmp, idx, results):
            if len(tmp) == k:
                if target == 0:
                    results.append(tmp)
                return
            
            for i in range(idx, 10):
                if i <= target:
                    dfs(k, target - i, tmp + [i], i+1, results)
        
        results = []
        dfs(k, n, [], 1, results)
        return results

    # 78. 子集, Medium
    def subsets(self, nums: List[int]) -> List[List[int]]:

        def dfs(nums, idx, tmp, results, k):
            if len(tmp) == k:
                results.append(tmp)
                return 
            

            for i in range(idx, len(nums)):
                dfs(nums, i+1, tmp+[nums[i]], results, k)
        
        results = []
        for i in range(len(nums)+1):
            dfs(nums, 0, [], results, i)
        
        return results

    # 90. 子集 II, Medium
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:

        def dfs(nums, idx, tmp, results, k):
            if len(tmp) == k:
                results.append(tmp)
                return 

            for i, num in enumerate(nums):
                if i > 0 and nums[i] == nums[i - 1]:
                    continue
                dfs(nums[i+1:], i + 1, tmp + [num], results, k)
        results = []
        nums.sort()
        for i in range(len(nums) + 1):
            dfs(nums, 0, [], results, i)
        return results

    # 131. 分割回文串, Medium
    def partition(self, s: str) -> List[List[str]]:

        def dfs(s, idx, tmp, results):
            if idx == len(s):
                results.append(tmp)
            
            for i in range(idx, len(s)):
                if s[idx:i+1] == s[idx:i+1][::-1]:
                    dfs(s, i+1, tmp+[s[idx:i+1]], results)
        results = []
        dfs(s, 0, [], results)
        return results

    # 37. 解数独, Hard
    def solveSudoku(self, board: List[List[str]]) -> None:
        def is_valid(board, x, y):
            tmp = board[x][y]
            board[x][y] = "."

            for row in range(9):
                if board[row][y] == tmp:
                    return False
            for col in range(9):
                if board[x][col] == tmp:
                    return False
            for row in range(3):
                for col in range(3):
                    if board[(x//3)*3+row][(y//3)*3+col] == tmp:
                        return False
            board[x][y] = tmp
            return True
                    
        def dfs(board):
            for r in range(9):
                for c in range(9):
                    if board[r][c] == ".":
                        for num in "123456789":
                            board[r][c] = num
                            if is_valid(board, r, c) and dfs(board):
                                return True
                            board[r][c] = "."
                        return False
            return True

        dfs(board)

    # 51. N 皇后, Hard
    def solveNQueens(self, n: int) -> List[List[str]]:
        
        def is_valid(board, x, y, n):

            board[x][y] = '.'

            for row in range(n):
                if board[row][y] == 'Q':  # 纵向
                    return False
            for col in range(n):
                if board[x][col] == 'Q':  # 横向
                    return False

            for row in range(n):
                col = row + (y - x)  # 主对角线
                if 0 <= col < n:
                    if board[row][col] == 'Q':
                        return False
                col = -row + (y + x)  # 副对角线
                if 0 <= col < n:
                    if board[row][col] == 'Q':
                        return False

            board[x][y] = 'Q'
            return True
        
        def dfs(board, n, step, results):

            if step == n:
                results.append(["".join(one_row) for one_row in board])
                return True

            for row in range(step, n):
                for col in range(n):
                    if board[row][col] == '.':
                        board[row][col] = 'Q'
                        if is_valid(board, row, col, n):
                            dfs(board, n, step+1, results)
                        board[row][col] = '.'
                return False
            return False

        results = []
        board = [["." for _ in range(n)] for _ in range(n)]
        flag = dfs(board, n, 0, results)
        return results

def main():
    s = Solution()

    # print(s.shortestPathBinaryMatrix([[0,0,0],[0,1,0],[0,0,0]]))
    # print(s.numSquares(12))

    # print(s.ladderLength(beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log","cog"]))

    # print(s.maxAreaOfIsland([[0,0,1,0,0,0,0,1,0,0,0,0,0],
    #                         [0,0,0,0,0,0,0,1,1,1,0,0,0],
    #                         [0,1,1,0,1,0,0,0,0,0,0,0,0],
    #                         [0,1,0,0,1,1,0,0,1,0,1,0,0],
    #                         [0,1,0,0,1,1,0,0,1,1,1,0,0],
    #                         [0,0,0,0,0,0,0,0,0,0,1,0,0],
    #                         [0,0,0,0,0,0,0,1,1,1,0,0,0],
    #                         [0,0,0,0,0,0,0,1,1,0,0,0,0]]))

    # print(s.numIslands(grid = [
    #                             ["1","1","0","0","0"],
    #                             ["1","1","0","0","0"],
    #                             ["0","0","1","0","0"],
    #                             ["0","0","0","1","1"]
    #                             ]))

    # print(s.findCircleNum([[1,1,0],[1,1,0],[0,0,1]]))

    # board = [["O","O","O","O"],["O","O","O","O"],["O","O","O","O"],["O","O","O","O"]]
    # print(tabulate(board))
    # s.solve(board)
    # print(tabulate(board))

    # print(s.pacificAtlantic(heights = [[1,2,2,3,5],[3,2,3,4,4],[2,4,5,3,1],[6,7,1,4,5],[5,1,1,2,4]]))

    # print(s.letterCombinations("234"))
    # print(s.restoreIpAddresses("25525511135"))

    # print(s.exist(board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCCED"))

    # tree = TreeNode(1)
    # tree.left = TreeNode(2)
    # tree.right = TreeNode(3)
    # tree.left.right = TreeNode(5)
    # print(s.binaryTreePaths(tree))

    # print(s.permute([1,2,3]))
    # print(s.permuteUnique([1,1,2]))
    print(s.combine(n=4,k=2))
    # print(s.combinationSum([2,3,5],8))
    # print(s.combinationSum3(k=3,n=9))
    # print(s.subsets([]))
    # print(s.subsetsWithDup([4,4,4,1,4]))
    # print(s.partition("a"))

    # board = [["5","3",".",".","7",".",".",".","."],["6",".",".","1","9","5",".",".","."],[".","9","8",".",".",".",".","6","."],["8",".",".",".","6",".",".",".","3"],["4",".",".","8",".","3",".",".","1"],["7",".",".",".","2",".",".",".","6"],[".","6",".",".",".",".","2","8","."],[".",".",".","4","1","9",".",".","5"],[".",".",".",".","8",".",".","7","9"]]

    # print(tabulate(board))
    # s.solveSudoku(board)
    # print(tabulate(board))

    # print(s.solveNQueens(4))


if __name__ == "__main__":
    main()
