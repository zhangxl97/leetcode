from typing import List
from tabulate import tabulate
from singly_linked_list import ListNode, print_nodes, connect_nodes
from tree import TreeNode


class Solution:
    # 91 Decode Ways, Medium
    # A -> 1 ... Z -> 26
    # "226" -> "BBF", "BZ", "VF" -> 3
    def numDecodings(self, s: str) -> int:
        size = len(s)
        if size == 1:
            return 1 if s != "0" else 0
        if s[0] == "0":
            return 0
        dp = [0 for _ in range(size + 1)]
        dp[0] = 1

        for i in range(1, size + 1):
            if s[i - 1] == "0":
                dp[i] = 0
            else:
                dp[i] = dp[i - 1]
            
            if i >= 2 and 10 <= int(s[i - 2: i]) <= 26:
                dp[i] += dp[i - 2]

        print(dp)
        return dp[-1]

    # 92 Reverse Linked List II, Medium
    def reverseBetween(self, head: ListNode, m: int, n: int) -> ListNode:
        # Input: 1->2->3->4->5->NULL, m = 2, n = 4
        # Output: 1->4->3->2->5->NULL
        if m == n or head is None or head.next is None:
            return head
        
        pre = ListNode(0)
        pre.next = head

        cnt = 0
        p = pre
        tmp1 = p.next
        while cnt < n - 1:
            if m - 1 <= cnt:
                if tmp1.next:
                    tmp2 = p.next
                    p.next = tmp1.next
                    tmp1.next = p.next.next
                    p.next.next = tmp2
                else:
                    pass
            else:
                p = p.next
                tmp1 = p.next
            cnt += 1
            # print_nodes(pre)
        return pre.next

    # 93 Restore IP Addresses， Medium
    def restoreIpAddresses(self, s: str) -> List[str]:
        size = len(s)
        if size <= 3:
            return []
        elif size == 4:
            return [".".join(list(s))]
        

        def helper(length, s, ips, result):
            if not s:
                if length == 4:
                    result.append('.'.join(ips))  # 以.分隔作为字符串返回
                return
            if length == 4:  # 分了4段，结束
                return

            # 取一位
            helper(length + 1, s[1:], ips + [s[:1]], result)

            # 若要取2位及以上，要确保目前的第一位不能为0
            if s[0] != '0':  
                if len(s) >= 2:
                    helper(length + 1, s[2:], ips + [s[:2]], result)
                if len(s) >= 3 and int(s[:3]) <= 255: # 若要取3位，则要保证小于255
                    helper(length + 1, s[3:], ips + [s[:3]], result)    

        result = []
        helper(0, s, [], result)
        return result

    # 94 Binary Tree Inorder Traversal, Medium
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        def helper(root, ans):
            if root:
                helper(root.left, ans)
                ans.append(root.val)
                helper(root.right, ans)

        ans = []
        helper(root, ans)
        return ans

    # 95 Unique Binary Search Trees II, Medium
    def generateTrees(self, n):
        def helper(start, end):
            result = []
            if start > end:
                result.append(None)
                return result

            for i in range(start, end + 1):
                # generate left and right sub tree
                leftTree = helper(start, i - 1)
                rightTree = helper(i + 1, end)
                # link left and right sub tree to root(i)
                for j in range(len(leftTree)):
                    for k in range(len(rightTree)):
                        root = TreeNode(i)
                        root.left = leftTree[j]
                        root.right = rightTree[k]
                        result.append(root)

            return result
        return helper(1, n)

    # 96 Unique Binary Search Trees, Medium
    def numTrees(self, n: int) -> int:
        # 以i为根，分别以[1,i-1],[i+1,n]为左右子树构造
        # 左右子树的节点个数分别从0变化至n-1，便能通过动态规划使用由n-1计算出的个数计算得出
        if n == 1:
            return 1
        dp = [0 for _ in range(n+1)]
        dp[0] = 1
        dp[1] = 1
        for i in range(2, n+1):
            for j in range(0, i):
                dp[i] += dp[j] * dp[i - j - 1]
        return dp[n]

    # 97 Interleaving String, Hard
    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        l1, l2, l3 = len(s1), len(s2), len(s3)
        if l1 + l2 != l3:
            return False

        dp = [[False for _ in range(l2 + 1)] for _ in range(l1 + 1)]

        for i1 in range(l1 + 1):
            for i2 in range(l2 + 1):
                if i1 == 0 and i2 == 0:
                    dp[i1][i2] = True
                elif i1 == 0:
                    dp[0][i2] = dp[0][i2 - 1] and s3[i2 - 1] == s2[i2 - 1]
                elif i2 == 0:
                    dp[i1][0] = dp[i1 - 1][0] and s3[i1 - 1] == s1[i1 - 1]
                else:
                    dp[i1][i2] = (dp[i1][i2 - 1] is True and s3[i1 + i2 - 1] == s2[i2 - 1]) or (dp[i1 - 1][i2] is True and s3[i1 + i2 -1] == s1[i1 - 1])
        # print(tabulate(dp))
        return dp[-1][-1]

    # 98 Validate Binary Search Tree, Medium
    def isValidBST(self, root: TreeNode) -> bool:
        if root is None:
            return True
        
        # order = []
        # def inOrder(root):
        #     if root is None:
        #         return 
            
        #     inOrder(root.left)
        #     order.append(root.val)
        #     inOrder(root.right)
        
        # inOrder(root)
        
        # return order == sorted(order) and len(set(order)) == len(order)

        # 如果左子树的值小于根的值并且右子树的值大于根的值，并进行递归，成立则为二叉搜索树，否则则不是。
        def helper(root, min, max):
            if root is None: 
                return True
            if min is not None and root.val <= min:
                return False
            if max is not None and root.val >= max:
                return False
            if helper(root.left,min,root.val) and helper(root.right,root.val,max):
                return True
            else:
                return False
        
        min_value, max_value = None, None
        return helper(root, min_value, max_value)

    # 99 Recover Binary Search Tree, Hard
    def recoverTree(self, root: TreeNode) -> None:
        # S(N) = O(1)算法：https://blog.csdn.net/qqxx6661/article/details/76565882  

        nodes, vals = [], []
        def helper(root):
            if root is None:
                return 
            
            helper(root.left)
            nodes.append(root)
            vals.append(root.val)
            helper(root.right)
        
        helper(root)
        tmp = [vals[i] - sorted(vals)[i] for i in range(len(vals))]
        recover = []
        for i, t in enumerate(tmp):
            if t != 0:
                recover.append(i)
        nodes[recover[0]].val = vals[recover[1]]
        nodes[recover[1]].val = vals[recover[0]]

    # 100 Same Tree, Easy
    def isSameTree(self, p: TreeNode, q: TreeNode) -> bool:
        if p == None and q == None:
            return True
        
        if p and q and p.val == q.val:
            return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)
        return False
 

def main():
    s = Solution()

    # 91
    # print(s.numDecodings("2611055971756562")) 
    # 92
    # nodes = connect_nodes([1,2,3,4,5])

    # print_nodes(nodes)
    # print_nodes(s.reverseBetween(nodes, 1, 5))
    # 93
    # print(s.restoreIpAddresses("0000000"))
    # 94
    # root = TreeNode(1)
    # root.right = TreeNode(2)
    # root.right.left = TreeNode(3)
    # print(s.inorderTraversal(root))
    # 95
    # for tree in s.generateTrees(3):
    #     tree.pre_order(tree)
    #     print("")
    # 96
    # print(s.numTrees(3))
    # 97
    # print(s.isInterleave(s1 = "db", s2 = "b", s3 = "cbb"))
    # 98
    # root = TreeNode(1)
    # root.left = TreeNode(3)
    # root.right = TreeNode(6)
    # root.right.left = TreeNode(3)
    # root.right.right = TreeNode(7)
    # print(s.isValidBST(root))
    # 99
    # root = TreeNode(1)
    # root.left = TreeNode(3)
    # root.left.right = TreeNode(2)
    # root.in_order(root)
    # print("")
    # s.recoverTree(root)
    # root.in_order(root)
    # 100
    root1 = TreeNode(1)
    root1.left = TreeNode(2)
    root1.right = TreeNode(1)

    root2 = TreeNode(1)
    root1.left = TreeNode(1)
    root2.right = TreeNode(2)
    print(s.isSameTree(root1, root2))


if __name__ == "__main__":
    main()
