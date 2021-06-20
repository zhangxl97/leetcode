from typing import List
from singly_linked_list import connect_nodes, print_nodes, ListNode


class Solution:
    # 160 Easy
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        l1 = headA
        l2 = headB
        while l1 != l2:
            if l1 is None:
                l1 = headB
            else:
                l1 = l1.next

            if l2 is None:
                l2 = headA
            else:
                l2 = l2.next

        return l1

    # 206, Easy
    def reverseList(self, head: ListNode) -> ListNode:
        ans = []
        p = head
        while p:
            ans.append(p.val)
            p = p.next
        
        p = head
        while p:
            p.val = ans.pop()
            p = p.next
        return head

    # 21, Easy
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        if l1 is None and l2 is None:
            return None
        elif l1 is None:
            return l2
        elif l2 is None:
            return l1
        
        head = ListNode(-1)
        p = head

        while l1 or l2:

            if l1 is None:
                p.next = ListNode(l2.val)
                l2 = l2.next
            elif l2 is None:
                p.next = ListNode(l1.val)
                l1 = l1.next
            else:
                if l1.val < l2.val:
                    p.next = ListNode(l1.val)
                    l1 = l1.next
                else:
                    p.next = ListNode(l2.val)
                    l2 = l2.next
            p = p.next
        return head.next

    # 83, Easy
    def deleteDuplicates(self, head: ListNode) -> ListNode:
        if head is None:
            return head
        
        p = head

        while p.next:
            p_next = p.next
            if p.val == p_next.val:
                while p_next.next and p_next.next.val == p.val:
                    p_next = p_next.next
                p.next = p_next.next
            else:
                p = p.next
        return head

    # 19, Medium
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        tmp = head
        for _ in range(n):
            tmp = tmp.next
        
        if tmp is None:
            return head.next

        p = head
        while tmp.next:
            tmp = tmp.next
            p = p.next
        
        p.next = p.next.next
        return head

    # 24. 两两交换链表中的节点, Medium
    def swapPairs(self, head: ListNode) -> ListNode:
        if head is None or head.next is None:
            return head
        
        pre = ListNode(-1)
        pre.next = head
        p = head
        p_next = head.next

        head = p_next

        while p and p_next:
            pre.next = p_next
            p.next = p_next.next
            p_next.next = p

            pre = p
            p = p.next
            p_next = p.next if p else None
        
        return head

    # 445 两数相加, Medium
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        n1 = 0
        n2 = 0
        p = l1
        while p:
            n1 = p.val + n1 * 10
            p = p.next
        p = l2
        while p:
            n2 = p.val + n2 * 10
            p = p.next
        ans = str(n1 + n2)
        head = ListNode(int(ans[0]))
        p = head
        for c in ans[1:]:
            p.next = ListNode(int(c))
            p = p.next
        return head
        
    # 234. 回文链表, Easy
    def isPalindrome(self, head: ListNode) -> bool:
        # nums = []
        # while head:
        #     nums.append(head.val)
        #     head = head.next
        # return nums == nums[::-1]
        pre = None
        slow = head
        fast = head  # fast以两倍速向后走，则等fast到链表尾部时，slow正好在中间
        while fast and fast.next:
            next_node = slow.next
            fast = fast.next.next

            slow.next = pre  # 反转节点
            pre = slow
            slow = next_node
        
        if fast:  # 有奇数个节点
            slow = slow.next
        
        while slow and pre:
            if slow.val != pre.val:
                return False
            slow = slow.next
            pre = pre.next
        return True

    # 725. 分隔链表, Medium
    def splitListToParts(self, root: ListNode, k: int) -> List[ListNode]:

        ans = [None] * k

        if root is None:
            return ans

        size = 0
        p = root
        while p:
            size += 1
            p = p.next
        
        N = size // k
        mod = size % k
        curr = root
        for i in range(k):
            size_i = N
            if mod > 0:
                size_i += 1
                mod -= 1
            
            ans[i] = curr
            for _ in range(size_i - 1):
                curr = curr.next
            next_node = curr.next
            curr.next = None
            curr = next_node
            if curr is None:
                break
        return ans

    # 328. 奇偶链表, Medium
    def oddEvenList(self, head: ListNode) -> ListNode:
        if head is None or head.next is None:
            return head
        
        odd, even = head, head.next
        even_head = even

        while even is not None and even.next is not None:
            odd.next = even.next
            even.next = even.next.next
            odd = odd.next
            even = even.next

        odd.next = even_head
        return head


def main():
    s = Solution()

    # headA = connect_nodes([4,1])
    # headB = connect_nodes([5,0,1])
    # headC = connect_nodes([8,4,5])
    # headA.next.next = headC
    # headB.next.next.next = headC
    # print_nodes(s.getIntersectionNode(headA, headB))

    # nodes = connect_nodes([1,2,3,4,5])
    # print_nodes(s.reverseList(nodes))

    # l1 = connect_nodes([])
    # l2 = connect_nodes([0])
    # print_nodes(s.mergeTwoLists(l1, l2))

    # nodes = connect_nodes([1,2,3,3])
    # print_nodes(s.deleteDuplicates(nodes))

    # nodes = connect_nodes([1,2,3,4,5])
    # print_nodes(nodes)
    # print_nodes(s.removeNthFromEnd(nodes, 5))

    # nodes = connect_nodes([])
    # print_nodes(nodes)
    # print_nodes(s.swapPairs(nodes))

    # l1 = connect_nodes([7,2,4,3])
    # l2 = connect_nodes([5,6,4])
    # print_nodes(s.addTwoNumbers(l1, l2))

    # nodes = connect_nodes([1,2,1])
    # print(s.isPalindrome(nodes))

    # nodes = connect_nodes([1])
    # ans = s.splitListToParts(nodes, k = 3)
    # for a in ans:
    #     print_nodes(a)
    #     # print("")

    nodes = connect_nodes([1,2,3,4,5,6])
    print_nodes(nodes)
    print_nodes(s.oddEvenList(nodes))

if __name__ == "__main__":
    main()

