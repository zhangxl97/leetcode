from typing import List


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


class Solution:
    # 21. Merge Two Sorted Lists
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        head = ListNode(0)
        p = head
        while l1 or l2:
            if l1 is None or (l2 is not None and l1.val >= l2.val):
                p.next = ListNode(l2.val)
                l2 = l2.next
                p = p.next
            elif l2 is None or (l1 is not None and l1.val < l2.val):
                p.next = ListNode(l1.val)
                l1 = l1.next
                p = p.next
        return head.next

    # 22 Generate Parentheses
    def dfs(self, res, word, left, right):
        if left == 0 and right == 0:
            res.append(word)

        if left > 0 and right >= left:
            self.dfs(res, word + '(', left - 1, right)
        if right > 0 and right - 1 >= left:
            self.dfs(res, word + ')', left, right - 1)

    def generateParenthesis(self, n: int) -> List[str]:
        res = []
        self.dfs(res, "", n, n)
        return res

    #     from itertools import product
    #     ans = []
    #     for i in product(['(', ')'], repeat=(n - 1) * 2):  # generate Full Permutation
    #         temp = ''.join(i)
    #         count = 0
    #         for ch in temp:
    #             if ch == '(':
    #                 count += 1
    #             else:
    #                 count -= 1
    #             if count < -1:
    #                 break
    #         if count == 0:
    #             ans.append('(' + ''.join(i) + ')')
    #     return ans
    # def generateParenthesis(self, n: int) -> List[str]:
    #     res = []
    #     self.dfs(n, n, "", res)
    #     return res
    #
    # def dfs(self, l, r, cur, res):
    #     if l == 0 and r == 0:
    #         res.append(cur)
    #     if l > 0 and r >= l:
    #         self.dfs(l-1, r, cur+'(', res)
    #     if r > 0 and r-1 >= l:
    #         self.dfs(l, r-1, cur+')', res)
    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        # size = len(lists)
        # keys = ['head_{}'.format(i) for i in range(len(lists))]
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

        elem = []
        for head in lists:
            p = head
            if p == []:
                break
            while p is not None:
                elem.append(p.val)
                p = p.next

        elem.sort()

        return connect_nodes(elem)

    # 24  Swap Nodes in Pairs
    def swapPairs(self, head: ListNode) -> ListNode:
        if head is None or head.next is None:
            return head

        res = head.next
        pre = head
        curr = pre.next
        next = curr.next

        pre.next = next
        curr.next = pre

        curr = pre.next
        next = None if curr is None else curr.next
        while next:
            pre.next = next
            temp = next.next
            next.next = curr
            curr.next = temp

            pre = curr
            curr = curr.next
            next = None if curr is None else curr.next
        return res

    # 25 Reverse Nodes in k-Group
    def reverseKGroup(self, head: ListNode, k: int) -> ListNode:
        if k == 1:
            return head
        if head is None:
            return None

        p = ListNode(0)
        p.next = head
        pre = p
        curr = pre.next
        if curr is None:
            return head
        next_p = curr
        for i in range(k - 1):
            next_p = next_p.next
            if next_p is None:
                return head

        while curr:

            pre.next = next_p
            curr_i = curr.next
            curr.next = next_p.next

            curr_pre = curr
            while curr_i != next_p:
                temp = curr_i.next
                curr_i.next = curr_pre
                curr_pre = curr_i
                curr_i = temp
            next_p.next = curr_pre

            pre = curr
            curr = curr.next
            next_p = curr
            if next_p is None:
                return p.next
            for i in range(k - 1):
                next_p = next_p.next
                if next_p is None:
                    return p.next
        return p.next

    # 26. Remove Duplicates from Sorted Array
    def removeDuplicates(self, nums: List[int]) -> int:
        res = list(set(nums))
        res.sort()
        nums[:len(res)] = res
        return len(res)

    # 27. Remove Element
    def removeElement(self, nums: List[int], val: int) -> int:
        for i in range(nums.count(val)):
            nums.remove(val)
        return len(nums)

    # 28 Implement strStr()
    def strStr(self, haystack: str, needle: str) -> int:
        # return haystack.find(needle)

        def same_start_end(s):
            """最长前后缀相同的字符位数"""
            n = len(s)  # 整个字符串长度
            j = 0  # 前缀匹配指向
            i = 1  # 后缀匹配指向
            result_list = [0] * n
            while i < n:
                if j == 0 and s[j] != s[i]:  # 比较不相等并且此时比较的已经是第一个字符了
                    result_list[i] = 0  # 值为０
                    i += 1  # 向后移动
                elif s[j] != s[i] and j != 0:  # 比较不相等,将j值设置为ｊ前一位的result_list中的值，为了在之前匹配到的子串中找到最长相同前后缀
                    j = result_list[j - 1]
                elif s[j] == s[i]:  # 相等则继续比较
                    result_list[i] = j + 1
                    j = j + 1
                    i = i + 1
            return result_list

        if needle == "":
            return 0
        """kmp算法,s是字符串，p是模式字符串，返回值为匹配到的第一个字符串的第一个字符的索引，没匹配到返回-1"""
        s_length = len(haystack)
        p_length = len(needle)
        i = 0  # 指向s
        j = 0  # 指向p
        next = same_start_end(needle)
        while i < s_length:
            if haystack[i] == needle[j]:  # 对应字符相同
                i += 1
                j += 1
                if j >= p_length:  # 完全匹配
                    return i - p_length
            elif haystack[i] != needle[j]:  # 不相同
                if j == 0:  # 与模式比较的是模式的第一个字符
                    i += 1
                else:  # 取模式当前字符之前最长相同前后缀的前缀的后一个字符继续比较
                    j = next[j - 1]

        if i == s_length:  # 没有找到完全匹配的子串
            return -1

    # 29. Divide Two Integers
    def divide(self, dividend: int, divisor: int) -> int:
        res = int(dividend / divisor)
        if res < -2 ** 31 or res > 2 ** 31 - 1:
            return 2 ** 31 - 1
        return res

    # 30. Substring with Concatenation of All Words
    def findSubstring(self, s: str, words: List[str]) -> List[int]:
        words_dict = {}
        word_num = len(words)
        for word in words:
            if word not in words_dict:
                words_dict[word] = 1
            else:
                words_dict[word] += 1
        word_len = len(words[0])
        res = []
        for i in range(len(s) + 1 - word_len * word_num):
            curr = {}
            j = 0
            while j < word_num:
                word = s[i + j * word_len:i + j * word_len + word_len]
                if word not in words:
                    break
                if word not in curr:
                    curr[word] = 1
                else:
                    curr[word] += 1
                if curr[word] > words_dict[word]: break
                j += 1
            if j == word_num:
                res.append(i)
        return res


def main():
    s = Solution()

    # 21
    # l1 = connect_nodes([1,4])
    # l2 = connect_nodes([1,3,4])
    # print_nodes(l1)
    # print_nodes(l2)
    # print_nodes(s.mergeTwoLists(l1, l2))
    # 22
    # print(s.generateParenthesis(2))
    # 23
    # l1 = connect_nodes([1, 4, 5])
    # l2 = connect_nodes([1, 3, 4])
    # l3 = connect_nodes([2, 6])
    # print_nodes(s.mergeKLists([])
    # 24
    # head = connect_nodes([1, 2, 3, 4, 5, 6])
    # print_nodes(s.swapPairs(head))
    # 25
    # head = connect_nodes([1,2,3,4,5,6])
    # print_nodes(s.reverseKGroup(head, 5))
    # 26
    # nums = [-1,0,0,0,0,3,3]
    # print(s.removeDuplicates(nums))
    # print(nums)
    # 27
    # nums = [3,2,2,3]
    # print(s.removeElement(nums, 2))
    # print(nums)
    # 28
    # print(s.strStr(haystack="hello", needle="helll"))
    # 29
    # print(s.divide(10, 3))
    # 30
    print(s.findSubstring(s="barfoofoobarthefoobarman", words=["bar", "foo", "the"]))


if __name__ == '__main__':
    main()
