from typing import List
from tree import TreeNode


class Solution:
    # 104. 二叉树的最大深度, Easy
    def maxDepth(self, root: TreeNode) -> int:
        if root is None:
            return 0
        return max(self.maxDepth(root.left), self.maxDepth(root.right)) + 1
    
    # 110. 平衡二叉树, Easy
    def isBalanced(self, root: TreeNode) -> bool:
        self.flag = True
        
        def max_depth(root):
            if root is None:
                return 0
            
            left = max_depth(root.left)
            right = max_depth(root.right)
            if abs(left - right) > 1:
                self.flag = False
                return
            return max(left, right) + 1

        max_depth(root)
        return self.flag

    # 543. 二叉树的直径, Easy
    def diameterOfBinaryTree(self, root: TreeNode) -> int:
        self.res = 0

        def check(root):
            if root is None:
                return 0
            
            left, right = check(root.left), check(root.right)

            self.res = max(left + 1 + right, self.res)
            return max(left, right) + 1
        
        check(root)
        return self.res - 1

    # 226. 翻转二叉树, Easy
    def invertTree(self, root: TreeNode) -> TreeNode:
        if root is None:
            return None

        left = root.left
        root.left = root.right
        root.right = left
        self.invertTree(root.left)
        self.invertTree(root.right)
        return root

    # 617. 合并二叉树, Easy
    def mergeTrees(self, root1: TreeNode, root2: TreeNode) -> TreeNode:

        def merge(root1, root2):
            if root1 is None:
                return root2
            if root2 is None:
                return root1
            
            root = TreeNode(root1.val + root2.val)
            root.left = merge(root1.left, root2.left)
            root.right = merge(root1.right, root2.right)
            return root
        
        root = merge(root1, root2)
        return root

    # 112. 路径总和, Easy
    def hasPathSum(self, root: TreeNode, targetSum: int) -> bool:

        # def check(root, targetSum):

        #     if root.left is None and root.right is None:
        #         if targetSum - root.val == 0:
        #             return True
        #         else:
        #             return False
        #     elif root.left is None:
        #         return check(root.right, targetSum - root.val)
        #     elif root.right is None:
        #         return check(root.left, targetSum - root.val)
        #     else:
        #         return check(root.left, targetSum - root.val) or check(root.right, targetSum - root.val)

        if root is None:
            return False
        if root.left is None and root.right is None and root.val == targetSum:
            return True
        return self.hasPathSum(root.left, targetSum - root.val) or self.hasPathSum(root.right, targetSum - root.val)
        
        # return check(root, targetSum)

    # 437. 路径总和 III, Medium
    def pathSum(self, root: TreeNode, targetSum: int) -> int:
        
        # def has_path(root, target):
        #     if root is None:
        #         return

        #     if target == root.val:
        #         self.num += 1

        #     left = has_path(root.left, target - root.val)
        #     right = has_path(root.right, target - root.val)
            
        # if root is None:
        #     return 0
        
        # queue = [root]
        # self.num = 0

        # while queue != []:
        #     curr = queue.pop(0)

        #     has_path(curr, targetSum)
        #     # print(curr.val, self.num)

        #     if curr.left:
        #         queue.append(curr.left)
        #     if curr.right:
        #         queue.append(curr.right)
        
        # return self.num

        def path_start_with_root(root, target):
            if root is None:
                return 0
            
            res = 0
            if root.val == target:
                res += 1
            
            res += path_start_with_root(root.left, target - root.val) + path_start_with_root(root.right, target - root.val)
            return res

        if root is None:
            return 0
        
        return path_start_with_root(root, targetSum) + self.pathSum(root.left, targetSum) + self.pathSum(root.right, targetSum)

    # 572. 另一个树的子树, Easy
    def isSubtree(self, s: TreeNode, t: TreeNode) -> bool:
        
        def check(s, t):
            if s is None and t is None:
                return True
            elif t is None or s is None:
                return False
            
            return s.val == t.val and check(s.left, t.left) and check(s.right, t.right)
        
        if t is None:
            return True
        if s is None:
            return False

        queue = [s]
         
        while queue != []:
            curr = queue.pop(0)

            if check(curr, t):
                return True
            
            if curr.left:
                queue.append(curr.left)
            if curr.right:
                queue.append(curr.right)
        return False


        if s is None:
            return False
        return check(s, t) or self.isSubtree(s.left, t) or self.isSubtree(s.right, t)

    # 101. 对称二叉树, Easy
    def isSymmetric(self, root: TreeNode) -> bool:
        def check(left, right):
            if left is None and right is None:
                return True
            if left is None or right is None:
                return False
            
            return left.val == right.val and check(left.left, right.right) and check(left.right, right.left)

        if root is None:
            return True
        return check(root.left, root.right)

    # 111. 二叉树的最小深度, Easy
    def minDepth(self, root: TreeNode) -> int:
        
        if root is None:
            return 0
        elif root.left is None:
            return self.minDepth(root.right) + 1
        elif root.right is None:
            return self.minDepth(root.left) + 1
        else:
            return min(self.minDepth(root.left), self.minDepth(root.right)) + 1

    # 404. 左叶子之和, Easy
    def sumOfLeftLeaves(self, root: TreeNode) -> int:
        self.res = 0

        # def check(root):
        #     if root is None:
        #         return
        #     if root.left is not None and root.left.left is None and root.left.right is None:
        #         self.res += root.left.val
        #         print(root.val, root.left.val)
            
        #     check(root.left)
        #     check(root.right)

        # check(root)
        # return self.res

        def is_leaf(root):
            if root is None:
                return 0
            return root.left is None and root.right is None

        if root is None:
            return 0
        if is_leaf(root.left):
            return root.left.val + self.sumOfLeftLeaves(root.right)
        
        return self.sumOfLeftLeaves(root.left) + self.sumOfLeftLeaves(root.right)

    # 687. 最长同值路径, Medium
    def longestUnivaluePath(self, root: TreeNode) -> int:
        def check(root):
            if root is None:
                # print(length)
                return 0

            left, right = check(root.left), check(root.right)

            if root.left and root.left.val == root.val:
                left_path = left + 1
            else:
                left_path = 0
            
            if root.right and root.right.val == root.val:
                right_path = right + 1
            else:
                right_path = 0
            self.path = max(self.path, left_path + right_path)
            return max(left_path, right_path)


        self.path = 0
        check(root)
        return self.path

    # 337. 打家劫舍 III, Medium
    def rob(self, root: TreeNode) -> int:
        # if root is None:
        #     return 0
        
        # val1 = root.val
        # if root.left:
        #     val1 += self.rob(root.left.left) + self.rob(root.left.right)
        # if root.right:
        #     val1 += self.rob(root.right.left) + self.rob(root.right.right)

        # val2 = self.rob(root.left) + self.rob(root.right)

        # return max(val1, val2)

        # res[2]  0 我们使用一个大小为 2 的数组来表示 int[] res = new int[2] 0 代表不偷，1 代表偷
        # 当前节点选择不偷：当前节点能偷到的最大钱数 = 左孩子能偷到的钱 + 右孩子能偷到的钱
        # 当前节点选择偷：当前节点能偷到的最大钱数 = 左孩子选择自己不偷时能得到的钱 + 右孩子选择不偷时能得到的钱 + 当前节点的钱数
        # 
        def rob_inter(root):
            if root is None:
                return [0, 0]
            
            res = [0, 0]
            left = rob_inter(root.left)
            right = rob_inter(root.right)

            res[0] = max(left[0], left[1]) + max(right[0], right[1]) 
            res[1] = left[0] + right[0] + root.val

            return res
        
        res = rob_inter(root)
        return max(res)

    # 671. 二叉树中第二小的节点, Easy
    def findSecondMinimumValue(self, root: TreeNode) -> int:
        if root is None:
            return -1

        min_value = float('inf')
        second_min = float('inf')

        queue = [root]
        while queue != []:
            curr = queue.pop(0)

            if curr.val < min_value:
                second_min = min_value
                min_value = curr.val
            elif curr.val == min_value:
                pass
            elif curr.val < second_min:
                second_min = curr.val
            
            if curr.left:
                queue.append(curr.left)
            if curr.right:
                queue.append(curr.right)

        return second_min if second_min != float('inf') else -1

    # 637. 二叉树的层平均值, Easy
    def averageOfLevels(self, root: TreeNode) -> List[float]:
        
        if root is None:
            return []

        res = []
        queue = [[root]]

        while queue != []:
            nodes = queue.pop()

            num = 0
            tmp = []
            for node in nodes:
                num += node.val
                if node.left:
                    tmp.append(node.left)
                if node.right:
                    tmp.append(node.right)
            res.append(num / len(nodes))
            if tmp != []:
                queue.append(tmp)
        return res

    # 513. 找树左下角的值, Medium
    def findBottomLeftValue(self, root: TreeNode) -> int:

        queue = [root]

        while queue != []:
            size = len(queue)

            left = None

            for i in range(size):
                node = queue.pop(0)
                if i == 0:
                    left = node.val
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)

            if queue == []:
                return left

    # 前序遍历 中左右
    def pre_order(self, root):
        if root:
            print(root.val)
            self.pre_order(root.left)
            self.pre_order(root.right)

    # 中序遍历 左中右
    def in_order(self, root):
        if root:
            self.in_order(root.left)
            print(root.val)
            self.in_order(root.right)
    
    # 后序遍历 左右中
    def post_order(self, root):
        if root:
            self.post_order(root.left)
            self.post_order(root.right)
            print(root.val)

    # 144. 二叉树的前序遍历, Easy
    def preorderTraversal(self, root: TreeNode) -> List[int]:
        if root is None:
            return []

        res = []
        stack = [root]

        while stack != []:

            node = stack.pop()
            res.append(node.val)
            if node.right:
                stack.append(node.right)
            if node.left:
                stack.append(node.left)

        return res

    # 145. 二叉树的后序遍历, Easy
    def postorderTraversal(self, root: TreeNode) -> List[int]:
        if root is None:
            return []

        res = []
        stack = [root]
        while stack != []:
            node = stack.pop()
            res.append(node.val)
            if node.left:
                stack.append(node.left)
            if node.right:
                stack.append(node.right)

        res.reverse()
        return res

    # 94. 二叉树的中序遍历, Medium
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        if root is None:
            return []

        res = []
        nodes = []
        curr = root
        while curr is not None or nodes != []:
            while curr is not None:
                nodes.append(curr)
                curr = curr.left
            
            node = nodes.pop()
            res.append(node.val)
            curr = node.right
        return res

    # 669. 修剪二叉搜索树, Medium
    def trimBST(self, root: TreeNode, low: int, high: int) -> TreeNode:
        if root is None:
            return None
        
        if root.val > high:
            return self.trimBST(root.left, low, high)
        elif root.val < low:
            return self.trimBST(root.right, low, high)
        
        left = self.trimBST(root.left, low, high)
        right = self.trimBST(root.right, low, high)
        root.left = left
        root.right = right
        return root

    # 230. 二叉搜索树中第K小的元素, Medium
    # BST中序遍历为升序
    def kthSmallest(self, root: TreeNode, k: int) -> int:
        def in_order(root):
            if root is None:
                return 
            in_order(root.left)
            self.cnt += 1
            if self.cnt == k:
                self.res = root.val
                return
            in_order(root.right)

        self.res = None
        self.cnt = 0
        in_order(root)
        return self.res
    
    # 538. 把二叉搜索树转换为累加树, Medium
    def convertBST(self, root: TreeNode) -> TreeNode:
        def travel(root):
            if root is None:
                return
            
            travel(root.right)
            self.sum += root.val
            root.val = self.sum
            travel(root.left)

        
        self.sum = 0
        travel(root)
        return root


def main():
    s = Solution()
    
    tree = TreeNode(4)

    tree.left = TreeNode(1)
    tree.right = TreeNode(6)

    tree.left.left = TreeNode(0)
    tree.left.right = TreeNode(2)
    tree.right.left = TreeNode(5)
    tree.right.right = TreeNode(7)

    # tree.left.left.left = TreeNode(-1)
    # tree.left.left.right = TreeNode(-2)
    # tree.left.right.left = TreeNode(1)
    tree.left.right.right = TreeNode(3)

    tree.right.right.right = TreeNode(8)

    # tree2 = TreeNode(4)
    # tree2.left = TreeNode(1)
    # tree2.right = TreeNode(2)
    # tree2.left.right = TreeNode(4)
    # tree2.right.right = TreeNode(7)

    # print(s.maxDepth(tree))
    # print(s.isBalanced(tree))
    # print(s.diameterOfBinaryTree(tree))
    # tree.BFS(tree)
    # print(s.invertTree(tree))
    # tree.BFS(tree)


    # tree.BFS(tree)
    # print("")
    # tree2.BFS(tree2)
    # print("")
    # root = s.mergeTrees(tree, tree2)
    # root.BFS(root)

    # print(s.hasPathSum(tree, 9))

    # print(s.pathSum(tree, -1))
    # print(s.isSubtree(tree, tree2))
    # print(s.isSymmetric(tree))
    # print(s.minDepth(tree))
    # print(s.sumOfLeftLeaves(tree))
    # print(s.longestUnivaluePath(tree))
    # print(s.rob(tree))
    # print(s.findSecondMinimumValue(tree))
    # print(s.averageOfLevels(tree))
    # print(s.findBottomLeftValue(tree))
    # print(s.pre_order(tree))
    # print(s.in_order(tree))
    # print(s.post_order(tree))
    # print(s.preorderTraversal(tree))
    # print(s.postorderTraversal(tree))
    # print(s.postorderTraversal(tree))
    # tree.pre_order(tree)
    # print("")
    # s.trimBST(tree, 1, 3)
    # tree.pre_order(tree)
    # tree.in_order(tree)
    # print(s.kthSmallest(tree, 2))
    print(s.convertBST(tree))


if __name__ == "__main__":
    main()
