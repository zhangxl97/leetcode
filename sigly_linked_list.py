
# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def connect_nodes(nums):
    if nums != []:
        head = ListNode()
        p = head
        for n in nums:
            p.next = ListNode(n)
            p = p.next
        return head.next
    else:
        return None


def print_nodes(head):
    # if head is not None:
    #     print(head.val, end="")
    while head is not None:
        print(head.val, end="")
        head = head.next
        if head is not None:
            print("->", end="")
    print("\n", end="")
