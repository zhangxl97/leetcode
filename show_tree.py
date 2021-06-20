# 
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

d_hor = 4   #节点水平距离
d_vec = 8   #节点垂直距离
radius = 2  #节点的半径

def get_lchild_width(root):
    '''获得根左边宽度'''
    return get_width(root.lchild)

def get_rchild_width(root):
    '''获得根右边宽度'''
    return get_width(root.rchild)

def get_width(root):
    '''获得树的宽度'''
    if root == None:
        return 0
    return get_width(root.lchild) + 1 + get_width(root.rchild)

def get_height(root):
    '''获得二叉树的高度'''
    if root == None:
        return 0
    return max(get_height(root.lchild), get_height(root.rchild)) + 1

def create_win(root):
    '''创建窗口'''
    WEIGHT, HEIGHT = get_w_h(root)
    WEIGHT = (WEIGHT+1)*d_hor
    HEIGHT = (HEIGHT+1)*d_vec
    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111)
    plt.xlim(0, WEIGHT)
    plt.ylim(0, HEIGHT)

    x = (get_lchild_width(root) + 1) * d_hor #x, y 是第一个要绘制的节点坐标，由其左子树宽度决定
    y = HEIGHT - d_vec
    return fig, ax, x, y     

def draw_a_node(x, y, data, ax):
    '''画一个节点'''
    c_node = Circle((x,y), radius=radius, color='gray')
    ax.add_patch(c_node)
    plt.text(x, y, '%d' % data, ha='center', va= 'bottom',fontsize=20)

def draw_a_edge(x1, y1, x2, y2, r=radius):
    '''画一条边'''
    x = (x1, x2)
    y = (y1, y2)
    plt.plot(x, y, 'k-')

def get_w_h(root):
    '''返回树的宽度和高度'''
    w = get_width(root)
    h = get_height(root)
    return w, h
def show_BTree(root):
    '''可视化二叉树'''
    _, ax, x, y = create_win(root)
    print_tree_by_inorder(root, x, y, ax)
    plt.show()

def print_tree_by_inorder(root, x, y, ax):
    '''通过中序遍历打印二叉树'''
    if root == None:
        return
    draw_a_node(x, y, root.data, ax)
    lx = rx = 0
    ly = ry = y - d_vec
    if root.lchild != None:
        lx = x - d_hor * (get_rchild_width(root.lchild) + 1)   #x-左子树的右边宽度
        draw_a_edge(x, y, lx, ly, radius)
    if root.rchild != None:
        rx = x + d_hor * (get_lchild_width(root.rchild) + 1)   #x-右子树的左边宽度
        draw_a_edge(x, y, rx, ry, radius)
    #递归打印    
    print_tree_by_inorder(root.lchild, lx, ly, ax)
    print_tree_by_inorder(root.rchild, rx, ry, ax)

def main():
    from BST import BST
    tree = BST([4,6,7,9,2,1,3,5,8])
    show_BTree(tree.get_root())

if __name__ == "__main__":
    main()
