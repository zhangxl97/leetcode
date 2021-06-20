from typing import List
from singly_linked_list import ListNode, connect_nodes, print_nodes
from tree import TreeNode, Node
from tabulate import tabulate


class Solution:
    # 111 Minimum Depth of Binary Tree, Easy
    def minDepth(self, root: TreeNode) -> int:

        if root == None:
            return 0
        if root.left == None and root.right != None:
            return self.minDepth( root.right ) + 1
        if root.left != None and root.right == None:
            return self.minDepth( root.left ) + 1
        return min( self.minDepth( root.left ), self.minDepth( root.right ) ) + 1
    
    # 112 Path Sum, Easy
    def hasPathSum(self, root: TreeNode, targetSum: int) -> bool:

        def helper(root, targetSum):
            targetSum -= root.val
            if root.left is None and root.right is None:
                if targetSum == 0:
                    return True
                else:
                    return False
            elif root.left is None:
                return helper(root.right, targetSum)
            elif root.right is None:
                return helper(root.left, targetSum)
            else:
                return helper(root.left, targetSum) or helper(root.right, targetSum)

        if root is None:
            return False
        else:
            return helper(root, targetSum)
    
    # 113 Path Sum II, Medium
    def pathSum(self, root: TreeNode, targetSum: int) -> List[List[int]]:
        from copy import deepcopy
        if root is None:
            return []

        ans = []

        def helper(root, targetSum, tmp):
            targetSum -= root.val
            tmp.append(root.val)
            if root.left is None and root.right is None:
                if targetSum == 0:
                    ans.append(deepcopy(tmp))
                else:
                    return 

            elif root.left is None:
                helper(root.right, targetSum, tmp)
                tmp.pop()
            elif root.right is None:
                helper(root.left, targetSum, tmp)
                tmp.pop()
            else:
                helper(root.left, targetSum, tmp)
                tmp.pop()
                helper(root.right, targetSum, tmp)
                tmp.pop()

        helper(root, targetSum, [])

        return ans
            
    # 114 Flatten Binary Tree to Linked List, Medium
    def flatten(self, root: TreeNode) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        def preOrder(root, pre):
            if root:
                pre.append(root.val)
                preOrder(root.left, pre)
                preOrder(root.right, pre)
        if root is None:
            return None
        nums = []
        preOrder(root, nums)
        
        p = root
        if p.left:
            p.left = None

        for num in nums[1:]:
            if p.left:
                p.left = None
            if p.right:
                p.right.val = num
            else:
                p.right = TreeNode(num)

            p = p.right

    # 115 Distinct Subsequences, Hard
    def numDistinct(self, s: str, t: str) -> int:
        if t == "":
            return 1
        elif s == "":
            return 0
            
        kv = {}
        for i, c in enumerate(s):
            if kv.get(c):
                kv[c].append(i)
            else:
                kv[c] = [i]

        currs = kv.get(t[0])
        if currs is None:
            return 0
        candidates = {c:1 for c in currs}
        # print(kv)

        # print(candidates)
        for c in t[1:]:
            currs = kv.get(c)
            if currs is None:
                return 0
            
            tmp = {}
            for curr in currs:
                cnt = 0
                for can in candidates.keys():
                    if curr > can:
                        cnt += candidates[can]
                if cnt > 0:
                    tmp[curr] = cnt
            candidates = tmp
            # print(c, end="\t")
            # print(candidates)


        ans = 0
        for can in candidates.keys():
            ans +=  candidates[can]
        return ans

    # 116 Populating Next Right Pointers in Each Node， Medium
    # 117 Populating Next Right Pointers in Each Node II, Medium
    def connect(self, root: 'Node') -> 'Node':
        if root is None:
            return None
        
        queue = []
        queue.append([root])

        while queue:
            curr_nodes = queue.pop(0)
            next_nodes = []
            size = len(curr_nodes)
            for i, node in enumerate(curr_nodes):
                if i == size - 1:
                    node.next = None
                else:
                    node.next = curr_nodes[i + 1]
                if node.left:
                    next_nodes.append(node.left)
                if node.right:
                    next_nodes.append(node.right)
            if next_nodes:
                queue.append(next_nodes)
        return root
                
    # 118 Pascal's Triangle, Easy
    def generate(self, numRows: int) -> List[List[int]]:
        ans = [[1]]
        for i in range(1, numRows):
            past = ans[i - 1]
            size = len(past)
            tmp = [1] * (size + 1)
            for j in range(1, size):
                tmp[j] = past[j - 1] + past[j]
            ans.append(tmp)
        return ans
    
    # 119 Pascal's Triangle II, Easy
    def getRow(self, rowIndex: int) -> List[int]:
        ans = [1]
        for i in range(1, rowIndex + 1):
            size = len(ans)
            tmp = [1] * (size + 1)
            for j in range(1, size):
                tmp[j] = ans[j - 1] + ans[j]
            ans = tmp
        return ans

    # 120 Triangle, Medium
    def minimumTotal(self, triangle: List[List[int]]) -> int:
        # dfs TLE Error
        # height = len(triangle)
        # if height == 1:
        #     return triangle[0][0]
        
        # def dfs(ans, row, col, height):
        #     if row == height:
        #         return ans
            
        #     ans += triangle[row][col]
        #     ans_left = dfs(ans, row+1, col, height)
        #     ans_right = dfs(ans, row+1, col+1, height)
        #     return min(ans_left, ans_right)
        
        # return dfs(0, 0, 0, height)

        # Dynamic Programming
        height = len(triangle)
        if height == 1:
            return triangle[0][0]
        dp = [[0 for j in range(i + 1)] for i in range(height)]

        dp[0][0] = triangle[0][0]
        for i in range(1, height):
            for j in range(i + 1):
                if j == 0:
                    dp[i][j] = dp[i - 1][j] + triangle[i][j]
                elif j == i:
                    dp[i][j] = dp[i - 1][j - 1] + triangle[i][j]
                else:
                    dp[i][j] = min(dp[i - 1][j], dp[i - 1][j - 1]) + triangle[i][j]
        # print(tabulate(dp))
        return min(dp[-1])


        print(tabulate(dp))



def main():
    s = Solution()

    # Tree

    root = TreeNode(1)

    root.left = TreeNode(2)
    root.right = TreeNode(5)

    # root.left.left = TreeNode(3)
    # root.left.right = TreeNode(4)
    # # root.right.left = TreeNode(13)
    # root.right.right = TreeNode(6)

    # root.left.left.left = TreeNode(7)
    # root.left.left.right = TreeNode(2)
    # root.right.right.left = TreeNode(5)
    # root.right.right.right = TreeNode(1)

    # 111
    # print(s.minDepth(root))
    # 112
    # print(s.hasPathSum(root, 3))
    # 113
    # print(s.pathSum(root, 22))
    # 114
    # root.pre_order(root)
    # print("")
    # root.in_order(root)
    # print("")
    # s.flatten(root)
    # root.pre_order(root)
    # print("")
    # root.in_order(root)

    # 115
    # print(s.numDistinct("rabbbits", "rabbits"))

    # 116, 117
    # print(s.connect(root))

    # 118
    # print(s.generate(5))

    # 119
    # print(s.getRow(3))
    # 120
    print(s.minimumTotal(triangle = [[-10]]))


if __name__ == "__main__":
    main()
