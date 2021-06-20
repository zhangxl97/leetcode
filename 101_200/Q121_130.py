from tabulate import tabulate
from singly_linked_list import ListNode, connect_nodes, print_nodes
from tree import TreeNode, Node
from typing import List


class Solution:
    # 121 Best Time to Buy and Sell Stock, Easy
    # one transaction
    def maxProfit1(self, prices: List[int]) -> int:
        size = len(prices)
        if size <= 1:
            return 0

        min_buy = prices[0]
        max_profit = 0
        # dp = [0 for _ in range(size)]
        # for i in range(1, size):
        #     profit = prices[i] - min_buy
        #     if profit > 0:
        #         dp[i] = profit
        #     if prices[i] < min_buy:
        #         min_buy = prices[i]
        # return max(dp)
        for i in range(1, size):
            if prices[i] > min_buy:
                profit = prices[i] - min_buy
                if profit > max_profit:
                    max_profit = profit
            else:
                min_buy = prices[i]
        return max_profit


    # 122 Best Time to Buy and Sell Stock II, Easy
    # multiple transactions 
    def maxProfit2(self, prices: List[int]) -> int:
        size = len(prices)
        if size <= 1:
            return 0

        min_buy = prices[0]
        tmp_max_profit = 0
        ans = 0
        for i in range(1, size):
            if prices[i] > min_buy:
                profit = prices[i] - min_buy
                if profit > tmp_max_profit:
                    tmp_max_profit = profit
                if i < size - 1 and prices[i] >= prices[i + 1]:
                    min_buy = prices[i + 1]
                    ans += tmp_max_profit
                    tmp_max_profit = 0

            else:
                min_buy = prices[i]

        ans += tmp_max_profit
        return ans

    # 123 Best Time to Buy and Sell Stock III, Hard
    # 2 transactions
    def maxProfit(self, prices: List[int]) -> int:
        days = len(prices)
        if days <= 1:
            return 0
        print(prices)
        dp = [[0 for _ in range(days)] for i in range(3)]  # 交易0，1，2次的总收益

        for i in range(1, 3):
            max_profit = -prices[0]
            for j in range(1, days):
                # TLE Error
                # local = 0
                # max_index = 0
                # for k in range(0, j):
                #     if prices[j] - prices[k] + dp[i - 1][k] > local:
                #         local = prices[j] - prices[k] + dp[i - 1][k]
                #         max_index = k
                # # dp[i][j - 1] 第i天什么都不做， 利润为前一天利润
                # # dp[i-1][j] + diff 第i天在局部最大利润时卖出
                # dp[i][j] = max(dp[i][j - 1], local)

                dp[i][j] = max(dp[i][j - 1], prices[j] + max_profit)
                max_profit = max(dp[i-1][j] - prices[j], max_profit)

        print(tabulate(dp))
        return dp[-1][-1]

    # 124 Binary Tree Maximum Path Sum, Hard
    def maxPathSum(self, root: TreeNode) -> int:
        def maxPathSum(root):
            if root is None:
                return 0
            
            left = maxPathSum(root.left)
            right = maxPathSum(root.right)

            self.maxSum = max(max(left, 0)+max(right, 0)+root.val, self.maxSum)

            # 返回的不是全局最大的路径和
            # 而是以当前节点为拐点的最大路径和，相当于局部最大路径和
            return max(left, right, 0) + root.val

        self.maxSum = float("-inf")
        maxPathSum(root)
        return self.maxSum
 

    # 125 Valid Palindrome, Easy
    def isPalindrome(self, s: str) -> bool:
        s = s.strip()
        s = s.lower()
        ans = ""
        for c in s:
            if "a" <= c <= "z" or "0" <= c <= "9":
                ans += c
        return ans == ans[::-1]

    # 126 word Ladders II, Hard
    def findLadders(self, beginWord: str, endWord: str, wordList: List[str]) -> List[List[str]]:
        def oneChange(word1, word2):
            return sum([word1[i] == word2[i] for i in range(len(word1))]) == len(word1) - 1

        if endWord not in wordList:
            return []
        from copy import deepcopy
        wordMap = {word: [] for word in wordList}
        wordMap[beginWord] = []
        size = len(wordList)
        for i in range(size):
            if wordList[i] == endWord:
                continue
            for j in range(size):
                if oneChange(wordList[i], wordList[j]):
                    if wordList[j] not in wordMap[wordList[i]]:
                        wordMap[wordList[i]].append(wordList[j])
                    if wordList[j] != endWord and wordList[i] not in wordMap[wordList[j]]:
                        wordMap[wordList[j]].append(wordList[i])
    
        for j in range(size):
            if oneChange(beginWord, wordList[j]) and wordList[j] not in wordMap[beginWord]:
                wordMap[beginWord].append(wordList[j])

        print(wordMap)
        print(beginWord, wordList)
        def dfs(word, tmp, ans, wordMap):
            
            if word == endWord and tmp not in ans:
                ans.append(deepcopy(tmp))
                return True
            for w in wordMap[word]:
                if w != word and w not in tmp:
                    tmp.append(w)
                    dfs(w, tmp, ans, wordMap)
                    # tmp = tmp[0:tmp.index(w)]
                    tmp.pop()

        ans = []
        dfs(beginWord, [beginWord], ans, wordMap)
        lengths = [len(tmp) for tmp in ans]
        res = []
        
        for a in ans:
            if len(a) == min(lengths):
                res.append(a)

        return res

    # 129 Sum Root to Leaf Numbers, Medium
    def sumNumbers(self, root: TreeNode) -> int:
        
        def _sumNumbers(root, preSum):
            if root is None:
                return 0
            
            preSum = preSum * 10 + root.val
            if root.left is None and root.right is None:
                return preSum

            return _sumNumbers(root.left, preSum) + _sumNumbers(root.right, preSum)
        
        return _sumNumbers(root, 0)

    # 130 Surrounded Regions, Medium
    def solve(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """

        rows = len(board)
        if rows <= 2:
            return
        cols = len(board[0])
        if cols <= 2:
            return
        
        def dfs(x, y):
            if x<0 or x>rows-1 or y<0 or y>cols-1 or board[x][y]!='O':
                return
            board[x][y] = 'D'
            dfs(x-1, y)
            dfs(x+1, y)
            dfs(x, y+1)
            dfs(x, y-1)
        
        for i in range(rows):
            dfs(i, 0); 
            dfs(i, cols-1)
        for j in range(1, cols-1):
            dfs(0, j); 
            dfs(rows-1, j)

        for i in range(rows):
            for j in range(cols):
                if board[i][j] == 'O':
                    board[i][j] = 'X'
                elif board[i][j] == 'D':
                    board[i][j] = 'O'


        # for i, pos in enumerate(flags.keys()):
        #     print("{}: {}.  ".format(pos, flags[pos]), end="")
        #     if (i + 1) % 4 == 0:
        #         print("")




def main():
    s = Solution()

    # tree
    # root = TreeNode(4)
    # root.left = TreeNode(9)
    # root.right = TreeNode(0)
    # root.left.left = TreeNode(5)
    # root.left.right = TreeNode(1)
    # root.right.left = TreeNode(-15)
    # root.right.right = TreeNode(7)

    # 121,122,123
    # print(s.maxProfit1(prices = [7,1,5,3,6,4]))
    # 124
    # print(s.maxPathSum(root))
    # 125
    # print(s.isPalindrome("0P"))

    # 126
    # print(s.findLadders(beginWord = "hot", endWord = "dog", wordList = ["hot","dog","dot"]))

    # 129 
    # print(s.sumNumbers(root))
    # 130
    # board = [["X","X","X","X","X"],["X","O","O","O","X"],["X","X","O","O","X"],["X","X","X","O","X"],["X","O","X","X","X"]]
    # print(tabulate(board))
    # print(s.solve(board))
    # print(tabulate(board))

    print(s.maxProfit(prices = [3,5,5,0,0,3,1,4]))



if __name__ == "__main__":
    main()
