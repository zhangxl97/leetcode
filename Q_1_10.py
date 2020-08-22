from typing import List


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    # Q1
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        from collections import defaultdict
        dic = defaultdict()
        for i, n in enumerate(nums):
            t = target - n
            if dic.get(t) is not None:
                return [dic[t], i]
            else:
                dic[n] = i

    # Q2
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        head = ListNode()
        p = head
        c = 0
        while l1 or l2:
            s = (0 if l1 is None else l1.val) + (0 if l2 is None else l2.val) + c
            p.next = ListNode(s % 10)
            p = p.next
            c = s // 10
            l1 = l1 if l1 is None else l1.next
            l2 = l2 if l2 is None else l2.next

        if c == 1:
            p.next = ListNode(1)

        return head.next

    # Q3
    def lengthOfLongestSubstring(self, s: str) -> int:
        position = {}
        max_length = 0
        start_pos = 0
        for i, c in enumerate(s):
            if position.get(c) is not None and position[c] >= start_pos:
                start_pos = position[c] + 1
            position[c] = i
            length = i - start_pos + 1
            if max_length < length:
                max_length = length

        return max_length

    # Q4
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        # nums = nums1
        # nums.extend(nums2)
        # nums = sorted(nums)
        # n = len(nums)
        # if n % 2 == 0:
        #     return (nums[n // 2 - 1] + nums[n // 2]) / 2.0
        # else:
        #     return nums[n // 2]

        n1 = len(nums1)
        n2 = len(nums2)
        n = n1 + n2

        if n % 2 == 1:
            if n1 == 0:
                return nums2[n // 2]
            elif n2 == 0:
                return nums1[n // 2]

            i = 0
            j = 0
            com = 0
            ans = 0
            while com < n // 2 + 1:
                if j >= n2 or (i < n1 and nums1[i] <= nums2[j]):
                    ans = nums1[i]
                    i += 1
                elif i >= n1 or (j < n2 and nums1[i] > nums2[j]):
                    ans = nums2[j]
                    j += 1
                com += 1
            return ans
        else:
            if n1 == 0:
                return (nums2[n // 2 - 1] + nums2[n // 2]) / 2.0
            elif n2 == 0:
                return (nums1[n // 2 - 1] + nums1[n // 2]) / 2.0
            i = 0
            j = 0
            com = 0
            ans = 0
            while com < n // 2:
                if j >= n2 or (i < n1 and nums1[i] <= nums2[j]):
                    ans = nums1[i]
                    i += 1
                elif i >= n1 or (j < n2 and nums1[i] > nums2[j]):
                    ans = nums2[j]
                    j += 1
                com += 1
            if i == n1:
                return (ans + nums2[j]) / 2.0
            elif j == n2:
                return (ans + nums1[i]) / 2.0
            else:
                return (ans + min(nums1[i], nums2[j])) / 2.0

    # Q5
    def longestPalindrome(self, s: str) -> str:
        if len(s) == 1 or s == s[::-1]:
            return s
        start_odd = 0
        start_even = 0
        max_odd = 1
        max_even = 0
        n = len(s)
        for i in range(n):
            odd = s[i - max_odd // 2 - 1: i + max_odd // 2 + 2]
            even = s[i - max_even: i + 2]
            while i - max_odd // 2 - 1 >= 0 and i + max_odd // 2 + 2 <= n and odd == odd[::-1]:
                start_odd = i - max_odd // 2 - 1
                max_odd += 2
                odd = s[i - max_odd // 2 - 1: i + max_odd // 2 + 2]
            while i - max_even >= 0 and i + 2 <= n and even == even[::-1]:
                start_even = i - max_even
                max_even += 2
                even = s[i - max_even: i + 2]
        if max_even < max_odd:
            return s[start_odd: start_odd + max_odd]
        else:
            return s[start_even: start_even + max_even]

    # 6. ZigZag Conversion
    def convert(self, s: str, numRows: int) -> str:
        if numRows == 1:
            return s
        import itertools
        dicts = {i: "" for i in range(numRows)}
        cycle = [i for i in range(numRows)]
        cycle.extend([i for i in range(numRows - 2, 0, -1)])
        cycle = itertools.cycle(cycle)
        for c in s:
            dicts[next(cycle)] += c
        return "".join(dicts.values())

    # 7. Reverse Integer
    def reverse(self, x: int) -> int:
        if x >= 0:
            x = int(str(x)[::-1])
            if x > 2 ** 31 - 1:
                return 0
            else:
                return x
        else:
            x = -int(str(x)[-1:0:-1])
            if x < - 2 ** 31:
                return 0
            else:
                return x

    # 8 String to Integer
    def myAtoi(self, str: str) -> int:
        str = str.strip()
        import re
        pattern = "^[+-]?[0-9]+"
        res = re.findall(pattern, str)
        try:
            res = int(res[0])
            if res > 2 ** 31 - 1:
                return 2 ** 31 - 1
            elif res < -2 ** 31:
                return -2 ** 31
            else:
                return res
        except:
            return 0

    # 9 Palindrome Number
    def isPalindrome(self, x: int) -> bool:
        if x < 0:
            return False
        return str(x) == str(x)[::-1]

    # 10 Regular Expression Mathing
    def isMatch(self, s: str, p: str) -> bool:
        import re
        res = re.findall("^" + p + "$", s)
        return res != []


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


def main():
    s = Solution()

    # Q1
    # nums = [2, 2, 4]
    # target = 4
    # print(s.twoSum(nums, target))
    # Q2
    # l1 = connect_nodes([5])
    # l2 = connect_nodes([5])
    # print_nodes(l1)
    # print_nodes(l2)
    # print_nodes(s.addTwoNumbers(l1, l2))
    # Q3
    # str = "abba"
    # print(s.lengthOfLongestSubstring(str))
    # Q4
    # nums1 = [2, 2, 2, 2]
    # nums2 = [2, 2, 2]
    # print(s.findMedianSortedArrays(nums1, nums2))
    # Q5
    # str = "ababad"
    # print(s.longestPalindrome(str))
    # Q6
    # print(s.convert("PAYPALISHIRING", 4))
    # Q7
    # print(s.reverse(12004564654351435435613514456))
    # Q8
    # print(s.myAtoi("-91283472332"))
    # Q9
    # print(s.isPalindrome(10))
    # Q10
    print(s.isMatch("mississippi", "mis*is*p*."))

if __name__ == '__main__':
    main()
