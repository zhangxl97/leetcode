# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Date: 2021/02/02

class BiTreeNode:
    def __init__(self, data):
        self.data = data
        self.lchild = None
        self.rchild = None
        self.parent = None


# Binary Search Tree
class BST:
    def __init__(self, nums=None):
        self.root = None
        if nums:
            for num in nums:
                self.insert_no_rec(num)

    def insert_rec(self, node, val):
        if node is None:
            node = BiTreeNode(val)
        
        elif val < node.data:
            node.lchild = self.insert(node.lchild, val)
            node.lchild.parent = node
        elif val > node.data:
            node.rchild = self.insert(node.rchild, val)
            node.rchild.parent = node
        else: # 相同键值不重复
            pass

        return node
    
    def insert_no_rec(self, val):
        p = self.root
        if p is None:
            self.root = BiTreeNode(val)  # 空树
            return
        
        while True:
            if val < p.data:
                if p.lchild:  # 左子树存在
                    p = p.lchild
                else:  # 左子树不存在
                    p.lchild = BiTreeNode(val)
                    p.lchild.parent = p
                    return 
            elif val > p.data:  # 右子树存在
                if p.rchild:
                    p = p.rchild
                else:  # 右子树不存在
                    p.rchild = BiTreeNode(val)
                    p.rchild.parent = p
                    return 
            else:
                return 

    def pre_order(self, node):
        if node:
            print(node.data, end=",")
            self.pre_order(node.lchild)
            self.pre_order(node.rchild)
    
    def in_order(self, node):
        if node:
            self.in_order(node.lchild)
            print(node.data, end=",")
            self.in_order(node.rchild)
    
    def post_order(self, node):
        if node:
            self.post_order(node.lchild)
            self.post_order(node.rchild)
            print(node.data, end=",")
    
    def __remove_node_1(self, node):
        # 情况1：node是叶子节点：直接删除
        # node为根节点
        if not node.parent: 
            self.root = None
    
        if node == node.parent.lchild:
            node.parent.lchild = None
            node.parent = None
        elif node == node.parent.rchild:
            node.parent.rchild = None
            node.parent = None
    
    def __remove_node_2_left(self, node):
        # 情况2_1: node只有一个左孩子：将其父节点与字子结点连起来。注意为根节点的情况，需更新根节点
        # node 为根节点
        if not node.parent:
            self.root = node.lchild
            node.lchild.parent = None
        elif node == node.parent.lchild:
            node.parent.lchild = node.lchild
            node.lchild.parent = node.parent
        elif node == node.parent.rchild:
            node.parent.rchild = node.lchild
            node.lchild.parent = node.parent
    
    def __remove_node_2_right(self, node):
        # 情况2_2: node只有一个右孩子。
        if not node.parent:
            self.root = node.rchild
            node.rchild.parent = None
        elif node == node.parent.lchild:
            node.parent.lchild = node.rchild
            node.rchild.parent = node.parent
        elif node == node.parent.rchild:
            node.parent.rchild = node.rchild
            node.rchild.parent = node.parent
        

    def delete(self, val):
        if self.root:  # 不是空树
            node = self.query_no_rec(val)
            if not node:  # 不存在该val
                return False
            
            if not node.lchild and not node.rchild:  # 1 叶子节点
                self.__remove_node_1(node)
            elif not node.rchild:  # 2_left 只有左孩子
                self.__remove_node_2_left(node)
            elif not node.lchild:  # 2_right 只有右孩子
                self.__remove_node_2_right(node)
            else:  
                # 3 两个孩子都有。将其右子树的最小节点（该节点最多有一个右孩子，即右子树的最左的节点）删除，并替换当前节点
                min_node = node.rchild
                while min_node.lchild:
                    min_node = min_node.lchild
                node.data = min_node.data  # 只替换数据
                # 删除min_node
                if min_node.rchild:  # 肯定没有左孩子，如果有右孩子，则满足 #2_right
                    self.__remove_node_2_right(min_node)
                else:  # min_node 为叶子节点
                    self.__remove_node_1(min_node)



    def query_no_rec(self, val):
        if self.root:
            p = self.root
            while p:
                if p.data == val:
                    return p
                elif p.data < val:
                    p = p.rchild
                else:
                    p = p.lchild
        return None


def main():
    # tree = BST([4,6,7,9,2,1,3,5,8])
    # tree.pre_order(tree.root)
    # print("")
    # tree.in_order(tree.root)  # BST的中序一定是升序
    # print("")
    # tree.post_order(tree.root)

    tree = BST([1,4,2,5,3,8,6,9,7])
    tree.in_order(tree.root)
    print("")
    tree.delete(4)
    tree.delete(1)
    tree.in_order(tree.root)



if __name__ == "__main__":
    main()
