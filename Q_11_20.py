from typing import List


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


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    # 11. Container With Most Water
    def maxArea(self, height: List[int]) -> int:
        n = len(height)
        if n == 2:
            return min(height)

        left = 0
        right = n - 1
        container = 0
        while left < right:
            temp = min(height[left], height[right]) * (right - left)
            if temp > container:
                container = temp
            if height[left] < height[right]:
                left += 1
            else:
                right -= 1
        return container

    # 12. Integer to Roman
    def intToRoman(self, num: int) -> str:
        res = ""
        while num > 0:
            if num >= 1000:
                res += "M" * (num // 1000)
                num %= 1000
            elif 900 <= num < 1000:
                res += "CM"
                num -= 900
            elif 500 <= num < 900:
                res += "D"
                num -= 500
            elif 400 <= num < 500:
                res += "CD"
                num -= 400
            elif 100 <= num < 400:
                res += "C" * (num // 100)
                num %= 100
            elif 90 <= num < 100:
                res += "XC"
                num -= 90
            elif 50 <= num < 90:
                res += "L"
                num -= 50
            elif 40 <= num < 50:
                res += "XL"
                num -= 40
            elif 10 <= num < 40:
                res += "X" * (num // 10)
                num %= 10
            elif 9 <= num < 10:
                res += "IX"
                num -= 9
            elif 5 <= num < 9:
                res += "V"
                num -= 5
            elif 4 <= num < 5:
                res += "IV"
                num -= 4
            else:
                res += "I" * num
                num = 0

        return res

    # 13. Roman to Integer
    def romanToInt(self, s: str) -> int:
        l = len(s) - 1
        dic = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
        ans = 0
        while l > 0:
            if s[l] == 'I':
                ans += 1
                l -= 1
            elif s[l] == 'V':
                if s[l - 1] == 'I':
                    ans += 4
                    l -= 2
                else:
                    ans += 5
                    l -= 1
            elif s[l] == 'X':
                if s[l - 1] == 'I':
                    ans += 9
                    l -= 2
                else:
                    ans += 10
                    l -= 1
            elif s[l] == 'L':
                if s[l - 1] == 'X':
                    ans += 40
                    l -= 2
                else:
                    ans += 50
                    l -= 1
            elif s[l] == 'C':
                if s[l - 1] == 'X':
                    ans += 90
                    l -= 2
                else:
                    ans += 100
                    l -= 1
            elif s[l] == 'D':
                if s[l - 1] == 'C':
                    ans += 400
                    l -= 2
                else:
                    ans += 500
                    l -= 1
            elif s[l] == 'M':
                if s[l - 1] == 'C':
                    ans += 900
                    l -= 2
                else:
                    ans += 1000
                    l -= 1
        if l == 0:
            ans += dic[s[0]]
        return ans

    # 14.
    def longestCommonPrefix(self, strs: List[str]) -> str:
        pre = ""
        try:
            n_str = len(strs)
            if n_str == 0:
                return ""
            elif n_str == 1:
                return strs[0]

            n = len(strs[0])
            for pos in range(n):
                temp = strs[0][pos]
                for num in range(1, n_str):
                    if strs[num][pos] == temp:
                        continue
                    else:
                        return pre
                pre += temp

        except:
            return pre
        return pre

    # 15 3Sum
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        # O(n^2)
        # size = len(nums)
        # if size < 3 or (size == 3 and sum(nums) != 0):
        #     return []
        # target = 0
        # nums.sort()
        # res = []
        # i = 0
        # while i < size - 2:
        #     t = target - nums[i]
        #     left = i + 1
        #     right = size - 1
        #     while left < right:
        #         if nums[left] + nums[right] == t:
        #             res.append([nums[i], nums[left], nums[right]])
        #             while nums[left] == nums[left + 1] and left < right - 1:
        #                 left += 1
        #             left += 1
        #             while nums[right] == nums[right - 1] and left < right - 1:
        #                 right -= 1
        #             right -= 1
        #         elif nums[left] + nums[right] < t:
        #             left += 1
        #         else:
        #             right -= 1
        #     while nums[i] == nums[i + 1] and i < size - 3:
        #         i += 1
        #     i += 1
        # return res

        # O(nlogn)
        from bisect import bisect_left, bisect_right
        from collections import Counter
        size = len(nums)
        if size < 3:
            return []
        res = []
        target = 0
        nums.sort()
        print(nums)
        count = Counter(nums)
        print(count)

        keys = list(count.keys())
        keys.sort()
        if target / 3 in keys and count[target / 3] >= 3:
            res.append([target // 3] * 3)

        begin = bisect_left(keys, target - keys[-1] * 2)
        end = bisect_left(keys, target * 3)
        for i in range(begin, end):
            a = keys[i]
            if count[a] >= 2 and target - 2 * a in count:
                res.append([a, a, target - 2 * a])

            max_b = (target - a) // 2  # target-a is remaining
            min_b = target - a - keys[-1]  # target-a is remaining and c can max be keys[-1]
            b_begin = max(i + 1, bisect_left(keys, min_b))
            b_end = bisect_right(keys, max_b)

            for j in range(b_begin, b_end):
                b = keys[j]
                c = target - a - b
                if c in count and b <= c:
                    if b < c or count[b] >= 2:
                        res.append([a, b, c])

        return res

    # 16 3sum cloest
    def threeSumClosest(self, nums: List[int], target: int) -> int:
        size = len(nums)
        if size == 3:
            return sum(nums)

        bias = float('inf')
        res = None
        nums.sort()
        i = 0
        while i < size - 2:
            left = i + 1
            right = size - 1
            while left < right:
                s = nums[i] + nums[left] + nums[right]
                if s == target:
                    return s
                elif abs(s - target) < bias:
                    res = s
                    bias = abs(s - target)
                if s < target:
                    while nums[left] == nums[left + 1] and left < right - 1:
                        left += 1
                    left += 1
                else:
                    while nums[right] == nums[right - 1] and left < right - 1:
                        right -= 1
                    right -= 1
            while nums[i] == nums[i + 1] and i < size - 3:
                i += 1
            i += 1
        return res

    # 17 Letter Combinations of a Phone Number
    def letterCombinations(self, digits: str) -> List[str]:
        kv = {'2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl', '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz'}
        size = len(digits)
        if size == 0:
            return []
        elif size == 1:
            return [c for c in kv[digits]]
        temp = []
        for i in range(size):
            temp.append(kv[digits[i]])

        words = []
        next_words = self.letterCombinations(digits[1:])
        for j in temp[0]:
            for i in next_words:
                words.append(j + i)
        return words

    # 18 4Sum
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        size = len(nums)
        if size < 4:
            return []
        elif size == 4:
            if sum(nums) != target:
                return []
            else:
                return [nums]

        nums.sort()
        res = []
        i = 0
        while i < size - 3:
            j = i + 1
            while j < size - 2:
                left = j + 1
                right = size - 1
                while left < right:
                    curr = nums[i] + nums[j] + nums[left] + nums[right]
                    if curr == target:
                        res.append([nums[i], nums[j], nums[left], nums[right]])
                        while nums[left] == nums[left + 1] and left < right - 1:
                            left += 1
                        left += 1
                        while nums[right] == nums[right - 1] and left < right - 1:
                            right -= 1
                        right -= 1
                    elif curr < target:
                        while nums[left] == nums[left + 1] and left < right - 1:
                            left += 1
                        left += 1
                    else:
                        while nums[right] == nums[right - 1] and left < right - 1:
                            right -= 1
                        right -= 1
                while nums[j] == nums[j + 1] and j < size - 3:
                    j += 1
                j += 1
            while nums[i] == nums[i + 1] and i < size - 4:
                i += 1
            i += 1
        return res

    # 19 Remove Nth Node From End of List
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        p = head
        size = 0
        while p is not None:
            size += 1
            p = p.next
        if size == n:
            return head.next
        elif size < n:
            return head
        p = head
        for i in range(size - n - 1):
            p = p.next
        temp = p.next
        p.next = temp.next
        temp.next = None
        del temp
        return head

    # 20 Valid Parentheses
    def isValid(self, s: str) -> bool:
        stack = []
        for c in s:
            if c == '(' or c == '[' or c == '{':
                stack.append(c)
            elif c == ')':
                if len(stack) > 0 and stack.pop() == '(':
                    continue
                else:
                    return False
            elif c == ']':
                if len(stack) > 0 and stack.pop() == '[':
                    continue
                else:
                    return False
            elif len(stack) > 0 and c == '}':
                if len(stack) > 0 and stack.pop() == '{':
                    continue
                else:
                    return False

        if stack == []:
            return True
        else:
            return False


def main():
    s = Solution()

    # 11
    # print(s.maxArea([1,100,6,2,100000,100000,8,3,7]))
    # 12
    # print(s.intToRoman(9))
    # 13
    # print(s.romanToInt("MCMXCIV"))
    # 14
    # print(s.longestCommonPrefix(["dog","racecar","car"]))
    # 15
    # print(s.threeSum([0,0,0]))
    # 16
    # print(s.threeSumClosest([0, 2, 1, -3], 1))
    # 17
    # print(s.letterCombinations("234"))
    # 18
    # print(s.fourSum([1, 0, -1, 0, -2, 2], 0))
    # 19
    # head = connect_nodes([1])
    # print_nodes(head)
    # head = s.removeNthFromEnd(head, 1)
    # print_nodes(head)
    # 20:
    print(s.isValid(")"))


if __name__ == '__main__':
    main()
