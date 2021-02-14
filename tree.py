# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

    def pre_order(self, root):
        # '''打印二叉树(先序)'''
		# 根左右
        if root == None:
            return 
        print(root.val, end=' ')
        self.pre_order(root.left)
        self.pre_order(root.right)

    def in_order(self, root):
        '''中序打印'''
        if root == None:
            return
        self.in_order(root.left)
        print(root.val, end=' ')
        self.in_order(root.right)

    def post_order(self, root):
        '''后序打印'''
        if root == None:
            return
        self.post_order(root.left)
        self.post_order(root.right)
        print(root.val, end=' ')

    def BFS(self, root):
        '''广度优先'''
        if root == None:
            return
        # queue队列，保存节点
        queue = []
        # res保存节点值，作为结果
        #vals = []
        queue.append(root)

        while queue:
            # 拿出队首节点
            currentNode = queue.pop(0)
            #vals.append(currentNode.val)
            print(currentNode.val, end=' ')
            if currentNode.left:
                queue.append(currentNode.left)
            if currentNode.right:
                queue.append(currentNode.right)
        #return vals

    def DFS(self, root):
        '''深度优先'''
        if root == None:
            return
        # 栈用来保存未访问节点
        stack = []
        # vals保存节点值，作为结果
        #vals = []
        stack.append(root)

        while stack:
            # 拿出栈顶节点
            currentNode = stack.pop()
            #vals.append(currentNode.val)
            print(currentNode.val, end=' ')
            if currentNode.right:
                stack.append(currentNode.right)
            if currentNode.left:
                stack.append(currentNode.left)            
        #return vals
        
    def maxDepth(self, root):
            """
            :type root: TreeNode
            :rtype: int
            """
            if not root:
                return 0
            left = self.maxDepth(root.left)+1
            right = self.maxDepth(root.right)+1
            
            return max(left, right)

    # Morris 遍历 S(N) = O(1)
    # https://blog.csdn.net/danmo_wuhen/article/details/104339630
    # https://blog.csdn.net/weixin_41665360/article/details/93055744
    def morris_pre_order(self, root):
        pass
    def morris_in_order(self, root):
        pass
    def morris_post_order(self, root):
        pass


def main():
	root = TreeNode(1)
	root.right = TreeNode(3)
	root.right.left = TreeNode(2)
	root.pre_order(root)


if __name__ == "__main__":
    main()

