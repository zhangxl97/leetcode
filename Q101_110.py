from tabulate import tabulate
from singly_linked_list import print_nodes, connect_nodes, ListNode
from tree import TreeNode
from typing import List


class Solution:
    # 101 Symmetric Tree, Easy
    def isSymmetric(self, root: TreeNode) -> bool:
        if root is None:
            return False
        def helper(left, right):
            if left is None and right is None:
                return True
           
            if left and right and left.val == right.val:
                return helper(left.left, right.right) and helper(left.right, right.left)
            return False
        
        return helper(root.left, root.right)

    # 102 Binary Tree Level Order Traversal, Medium
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        if root is None:
            return []
        
        queue = []
        queue.append([root])
        ans = []
        while queue:
            curr = queue.pop(0)
            ans_tmp = []
            node_tmp = []

            for node in curr:
                ans_tmp.append(node.val)
                if node.left:
                    node_tmp.append(node.left)
                if node.right:
                    node_tmp.append(node.right)

            if node_tmp:
                queue.append(node_tmp)
            ans.append(ans_tmp)
        return ans

    # 103 Binary Tree Zigzag Level Order Traversal, Medium
    def zigzagLevelOrder(self, root: TreeNode) -> List[List[int]]:
        if root is None:
            return []

        queue = []
        queue.append([root])
        ans = []
        flag = True

        while queue:
            curr_nodes = queue.pop(0)
            vals = []
            next_nodes = []
            for node in curr_nodes:
                vals.append(node.val)
                if node.right:
                    next_nodes.append(node.right)
                if node.left:
                    next_nodes.append(node.left)

            if next_nodes:
                queue.append(next_nodes)
            if flag:  # 奇数层为正序
                vals.reverse()
            flag = not flag
            ans.append(vals)
            
        return ans

    # 104 Maximum Depth of Binary Tree, Easy
    def maxDepth(self, root: TreeNode) -> int:
        if root is None:
            return 0
        
        def helper(root, height):
            if root is None:
                return height
            
            left = helper(root.left, height + 1)
            right = helper(root.right, height + 1)
            return max(left, right)
        
        return helper(root, 0)

    # 105 Construct Binary Tree from Preorder and Inorder Traversal, Medium
    def buildTree_pre_in(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        if preorder == []:
            return None
    
        # 先序遍历第一个肯定是根节点
        root = TreeNode(preorder[0])
        index = inorder.index(preorder[0])
        root.left = self.buildTree_pre_in(preorder[1:index+1], inorder[:index])
        root.right = self.buildTree_pre_in(preorder[index+1:], inorder[index+1:])
        return root
    
    # 106 Construct Binary Tree from inorder and Postorder Traversal, Medium
    def buildTree(self, inorder: List[int], postorder: List[int]) -> TreeNode:
        if inorder:
            # 后序遍历最后一个肯定为根节点
            root = TreeNode(postorder[-1])

            index = inorder.index(postorder[-1])
            root.left = self.buildTree(inorder[:index], postorder[:index])
            root.right = self.buildTree(inorder[index+1:], postorder[index:-1])

            return root
        else:
            return None

    # 107 Binary Tree Level Order Traversal II, Medium
    def levelOrderBottom(self, root: TreeNode) -> List[List[int]]:
        if root is None:
            return []
        
        queue = []
        ans = []
        
        queue.append([root])
        
        while queue:
            curr_nodes = queue.pop(0)
            
            vals = []
            next_nodes = []
            for node in curr_nodes:
                vals.append(node.val)
                if node.left:
                    next_nodes.append(node.left)
                if node.right:
                    next_nodes.append(node.right)
            if next_nodes:
                queue.append(next_nodes)
            ans.append(vals)
        
        ans.reverse()
        return ans

    # 108 Convert Sorted Array to Binary Search Tree, Easy
    def sortedArrayToBST(self, nums: List[int]) -> TreeNode:
        size = len(nums)
        if size == 0:
            return None
        elif size == 1:
            root = TreeNode(nums[0])
            return root

        mid = len(nums) // 2

        root = TreeNode(nums[mid])

        root.left = self.sortedArrayToBST(nums[:mid])
        root.right = self.sortedArrayToBST(nums[mid+1:])
        return root
    
    # 109 Convert Sorted List to Binary Search Tree, Medium
    def sortedListToBST(self, head: ListNode) -> TreeNode:
        p = head
        nums = []
        while p:
            nums.append(p.val)
            p = p.next
        
        def helper(nums: List[int]) -> TreeNode:
            size = len(nums)
            if size == 0:
                return None
            elif size == 1:
                root = TreeNode(nums[0])
                return root

            mid = len(nums) // 2

            root = TreeNode(nums[mid])

            root.left = helper(nums[:mid])
            root.right = helper(nums[mid+1:])
            return root

        return helper(nums)

    # 110 Balanced Binary Tree, Easy
    def isBalanced(self, root: TreeNode) -> bool:
        
        def helper(root, height):
            if root is None:
                return height, True
            
            left, left_flag = helper(root.left, height+1)
            right, right_flag = helper(root.right, height+1)
            return max(left, right), left_flag and right_flag and abs(right - left) <= 1
        
        height, flag = helper(root, 0)
        print(height)
        return flag



def main():
    s = Solution()

    # Tree

    root = TreeNode(1)
    # root.left = TreeNode(2)
    root.right = TreeNode(2)
    # root.left.left = TreeNode(3)
    root.right.right = TreeNode(3)
    # root.left.left.left = TreeNode(4)
    # root.left.left.right = TreeNode(4)


    # 101
    # print(s.isSymmetric(root))

    # 102
    # print(s.levelOrder(root))

    # 103
    # print(s.zigzagLevelOrder(root))

    # 104
    # print(s.maxDepth(root))

    # 105
    # tree = s.buildTree(preorder = [3,9,20,15,7], inorder = [9,3,15,20,7])
    # tree.pre_order(tree)
    # print("")
    # tree.in_order(tree)

    # 106
    # tree = s.buildTree(inorder = [9,3,15,20,7], postorder = [9,15,7,20,3])
    # tree.in_order(tree)
    # print("")
    # tree.post_order(tree)

    # 107
    # print(s.levelOrderBottom(root))

    # 108
    # tree = s.sortedArrayToBST([0,1,2,3,4,5])
    # tree.in_order(tree)
    # print("")
    # tree.pre_order(tree)

    # 109
    # nodes = connect_nodes([-10,-3,0,5,9])
    # tree = s.sortedListToBST(nodes)
    # tree.pre_order(tree)
    # print("")
    # tree.in_order(tree)

    # 110
    print(s.isBalanced(root))



if __name__ == "__main__":
    main()
