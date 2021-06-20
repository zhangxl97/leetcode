# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Date: 2021/02/02
from show_tree import show_BTree

class BiTreeNode:
    def __init__(self, data):
        self.data = data
        self.lchild = None
        self.rchild = None
        self.parent = None
        
    def is_leaf(self):
        return self.lchild is None and self.rchild is None
    
    def has_two_children(self):
        return self.lchild and self.rchild

    def is_left_child(self):
        return self.parent is not None and self == self.parent.lchild
    def is_right_child(self):
        return self.parent is not None and self == self.parent.rchild

class AVLNode(BiTreeNode):
    def __init__(self, data):
        super(AVLNode, self).__init__(data)
        self.height = 1
    
    def balance_factor(self):
        left_height = 0 if self.lchild is None else self.lchild.height
        right_height = 0 if self.rchild is None else self.rchild.height
        return left_height - right_height 

    def update_height(self):
        left_height = 0 if self.lchild is None else self.lchild.height
        right_height = 0 if self.rchild is None else self.rchild.height
        self.height = 1 + max(left_height, right_height)

    def taller_child(self):
        left_height = 0 if self.lchild is None else self.lchild.height
        right_height = 0 if self.rchild is None else self.rchild.height
        if left_height > right_height:
            return self.lchild
        elif left_height < right_height:
            return self.rchild
        else:
            # 如果两个子树一样高，则按照原来的方向返回
            # 即如果自己是父节点的左子树，则返回左子树；如果自己是父节点的右子树，则返回右子树。
            return self.lchild if self.is_left_child() else self.rchild


# Binary Search Tree
class BST:
    def __init__(self, nums=None):
        self.root = None
        if nums:
            for num in nums:
                self.insert_no_rec(num)

    def get_root(self):
        return self.root

    def insert_rec(self, node, val):
        
        if node is None:
            node = BiTreeNode(val)
        
        elif val < node.data:
            node.lchild = self.insert_rec(node.lchild, val)
            node.lchild.parent = node
        elif val > node.data:
            node.rchild = self.insert_rec(node.rchild, val)
            node.rchild.parent = node
        else: # 相同键值不重复
            pass

        return node
    
    def insert_no_rec(self, val):
        new_node = self.create_node(val) 

        p = self.root
        if p is None:
            self.root = new_node  # 空树
            return
        
        while True:
            if val < p.data:
                if p.lchild:  # 左子树存在
                    p = p.lchild
                else:  # 左子树不存在
                    p.lchild = new_node
                    new_node.parent = p
                    break
            elif val > p.data:  # 右子树存在
                if p.rchild:
                    p = p.rchild
                else:  # 右子树不存在
                    p.rchild = new_node
                    new_node.parent = p
                    break 
            else:
                break 
        
        self.after_add(new_node)

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
            # node.parent = None
        elif node == node.parent.rchild:
            node.parent.rchild = None
            # node.parent = None
    
    def __remove_node_2_left(self, node):
        # 情况2_1: node只有一个左孩子：将其父节点与字子结点连起来。注意为根节点的情况，需更新根节点
        # node 为根节点
        if node.parent is None:
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
        if node.parent is None:
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
                self.after_remove(node)
            elif node.rchild is None:  # 2_left 只有左孩子
                self.__remove_node_2_left(node)
                self.after_remove(node)
            elif node.lchild is None:  # 2_right 只有右孩子
                self.__remove_node_2_right(node)
                self.after_remove(node)
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
                    
                self.after_remove(min_node)
            

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

    def create_node(self, val):
        return BiTreeNode(val)
    # 添加node之后的调整
    def after_add(self, node):
        pass

    def after_remove(self, node):
        pass


class AVLTree(BST):
    def after_add(self, node):
        while node.parent:
            node = node.parent
            # node 是否平衡
            # 若平衡
            if self.is_balanced(node):
                # 更新高度
                self.update_height(node)
            # 若不平衡
            else: 
                # 恢复平衡，更新高度
                # 找到第一个高度最低的不平衡的节点（g），进行旋转操作
                # 结束后整棵树就平衡了
                self.rebalance(node)
                break

    def after_remove(self, node):
        while node.parent:
            node = node.parent 
            if self.is_balanced(node):
                self.update_height(node)
            else:
                self.rebalance(node)

    def create_node(self, val):
        return AVLNode(val)
    
    def is_balanced(self, node):
        return abs(node.balance_factor()) <= 1
    
    def update_height(self, node):
        node.update_height()
    
    def after_rotate(self, grand, parent, child):
        # 让parent成为子树根节点
        parent.parent = grand.parent
        if grand.is_left_child():
            grand.parent.lchild = parent
        elif grand.is_right_child():
            grand.parent.rchild = parent
        else:
            self.root = parent

        # 更新rchild
        if child:
            child.parent = grand

        #更新grand
        grand.parent = parent

        self.update_height(grand)
        self.update_height(parent)

    def rotate_right(self, grand):
        parent = grand.lchild
        rchild = parent.rchild

        grand.lchild = rchild
        parent.rchild = grand
        self.after_rotate(grand, parent, rchild)

    def rotate_left(self, grand):
        parent = grand.rchild
        lchild = parent.lchild

        grand.rchild = lchild
        parent.lchild = grand

        self.after_rotate(grand, parent, lchild)

    # 恢复平衡
    # param@ grand 高度最低的不平衡节点
    def rebalance_2(self, grand):
        parent = grand.taller_child()
        node = parent.taller_child()   

        if parent.is_left_child():  # L
            if node.is_left_child():  # LL
                self.rotate_right(grand)
            else:  # LR
                self.rotate_left(parent)
                self.rotate_right(grand)
        else:  # R
            if node.is_left_child():  # RL
                self.rotate_right(parent)
                self.rotate_left(grand)
            else:
                self.rotate_left(grand )

    # 恢复平衡
    # 统一所有旋转操作
    def rebalance(self, grand):
        
        parent = grand.taller_child()
        node = parent.taller_child()   

        if parent.is_left_child():  # L
            if node.is_left_child():  # LL
                self.rotate(grand, node.lchild, node, node.rchild, parent, parent.rchild, grand, grand.rchild)
            else:  # LR
                self.rotate(grand, parent.lchild, parent, node.lchild, node, node.rchild, grand, grand.rchild)
        else:  # R
            if node.is_left_child():  # RL
                self.rotate(grand, grand.lchild, grand, node.lchild, node, node.rchild, parent, parent.rchild)
            else:  # RR
                self.rotate(grand, grand.lchild, grand, parent.lchild, parent, node.lchild, node, node.rchild)

    def rotate(self, 
                    r,  # 子树根节点
                    a, b, c, 
                    d, 
                    e, f, g):

        # 让d成为子树根节点
        d.parent = r.parent
        if r.is_left_child():
            r.parent.lchild = d
        elif r.is_right_child():
            r.parent.rchild = d
        else:
            self.root = d
        
        # a-b-c
        b.lchild = a
        if a is not None:
            a.parent = b
        b.rchild = c
        if c is not None:
            c.parent = b
        self.update_height(b)

        # e-f-g
        f.lchild = e
        if e is not None:
            e.parent = f
        f.rchlid = g
        if g is not None:
            g.parent = f
        self.update_height(f)

        # b-d-f
        d.lchild = b
        d.rchild = f
        b.parent = d
        f.parent = d
        self.update_height(d)

def main():
    # tree = BST([4,6,7,9,2,1,3,5,8])
    # tree.pre_order(tree.root)
    # print("")
    # tree.in_order(tree.root)  # BST的中序一定是升序
    # print("")
    # tree.post_order(tree.root)

    # tree = BST([1,4,2,5,3,8,6,9,7])
    # tree.in_order(tree.root)
    # print("")
    # tree.delete(4)
    # tree.delete(1)
    # tree.in_order(tree.root)
    # print("")

    # bst = BST([4,6,7,9,2,1,3,5,8])
    # bst = BST([45, 92, 6, 57, 91, 31, 78, 51, 79, 13, 8, 35, 86])
    # bst.in_order(bst.get_root())
    # show_BTree(bst.get_root())
    # print("")

    # AVL = AVLTree([4,6,7,9,2,1,3,5,8])
    AVL = AVLTree([7, 1, 2, 3, 4, 5, 6, 8, 9, 10])
    AVL.in_order(AVL.get_root())
    print("")
    show_BTree(AVL.get_root())
    # AVL.delete(45)
    # AVL.delete(6)
    # show_BTree(AVL.get_root())
    # AVL.in_order(AVL.get_root())
    # print("")


if __name__ == "__main__":
    main()
