from typing import List
from singly_linked_list import ListNode


class Solution:
    # 345. 反转字符串中的元音字母, Easy
    def reverseVowels(self, s: str) -> str:
        s = list(s)
        left, right = 0, len(s) - 1
        vowels = "aeiouAEIOU"


        while left < right:
            if s[left] not in vowels:
                left += 1
                continue
            if s[right] not in vowels:
                right -= 1
                continue
            s[left], s[right] = s[right], s[left]
            left += 1
            right -= 1
            
        return "".join(s)

    # 680. 验证回文字符串 Ⅱ, Easy
    def validPalindrome(self, s: str) -> bool:
        left, right = 0, len(s) - 1
        while left < right:
            if s[left] == s[right]:
                left += 1
                right -= 1
            else:
                return s[left+1:right+1] == s[left+1:right+1][::-1] or s[left:right] == s[left:right][::-1]
        return True

    # 88. 合并两个有序数组, Easy
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        # j = 0
        # for i in range(m + n):
        #     if j >= n:
        #         break
        #     elif nums2[j] < nums1[i]:
        #         nums1[i + 1 : m + j + 1]  = nums1[i : m + j]
        #         nums1[i] = nums2[j]
        #         j += 1
        #     elif i >= m + j:
        #         nums1[i] = nums2[j]
        #         j += 1
        index1, index2 = m - 1, n - 1
        merge_index = m + n - 1

        while index1 >= 0 or index2 >= 0:
            if index1 < 0:
                nums1[merge_index] = nums2[index2]
                index2 -= 1
            elif index2 < 0:
                nums1[merge_index] = nums1[index1]
                index1 -= 1
            else:
                if nums2[index2] > nums1[index1]:
                    nums1[merge_index] = nums2[index2]
                    index2 -= 1
                else:
                    nums1[merge_index] = nums1[index1]
                    index1 -= 1
            merge_index -= 1

    # 141. 环形链表, Easy
    def hasCycle(self, head: ListNode) -> bool:
        if head is None:
            return False
        
        fast, slow = head, head
        while slow and fast and fast.next:
            fast = fast.next.next
            slow = slow.next
            if fast == slow:
                return True
        return False

    # 524. 通过删除字母匹配到字典里最长单词, Medium
    def findLongestWord(self, s: str, dictionary: List[str]) -> str:
        
        max_len = 0
        ans = ""
        for word in dictionary:
            l1 = 0
            l2 = 0
            len_word = len(word)
            while l2 < len_word and l1 < len(s):
                if s[l1] == word[l2]:
                    l1 += 1
                    l2 += 1
                else:
                    l1 += 1
            if l2 == len_word:
                if max_len < len_word or (max_len == len_word and word < ans):
                    max_len = len_word
                    ans = word
        return ans
                


def main():
    s = Solution()

    # print(s.reverseVowels("aA"))
    # print(s.validPalindrome("abca"))
    # nums1 = [1,2,3,0,0,0]
    # nums2 = [2,5,6]
    # s.merge(nums1, 3, nums2, 3)
    # print(nums1)

    # head = ListNode(1)
    # head.next = ListNode(2)
    # head.next.next = head
    # print(s.hasCycle(head))  

    print(s.findLongestWord("abce", ["abe","abc"]))

if __name__ == "__main__":
    main()
