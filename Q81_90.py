from typing import List
from tabulate import tabulate
from sigly_linked_list import ListNode, print_nodes, connect_nodes


class Solution:
    # 81. Search in Rotated Sorted Array II, Medium
    def search(self, nums: List[int], target: int) -> bool:
        size = len(nums)
        if size == 0:
            return False
        elif size == 1:
            return nums[0] == target

        index = 0
        while index < size:
            if nums[index] == target:
                return True
            # 当前比target小
            elif nums[index] < target:
                # 已经到rotated的末尾 e.g. 2,5,6,0,0,1,2的6
                if index < size - 1 and nums[index] > nums[index + 1]:
                    return False
                # 跳过相同的数
                while index < size - 1 and nums[index] == nums[index + 1]:
                    index += 1
                index += 1
            # 当前比target大
            else:
                # 跳转到rotated的下一个 e.g. 2,5,6,0,0,1,2的0
                while index < size - 1 and nums[index] <= nums[index + 1]:
                    index += 1
                index += 1
        return False

    # 82. Remove Duplicates from Sorted List II, Medium
    def deleteDuplicates(self, head: ListNode) -> ListNode:
        if head is None:
            return head

        start_point = ListNode(0)
        start_point.next = head

        start, end = start_point, head
        past = head.val
        p = head.next
        while p:
            now = p.val
            if now != past:
                past = now
                start = end
                end = p
                p = p.next
            else:
                p = p.next
                while p:
                    if p.val == past:
                        p = p.next
                    else:
                        break
                end = p
                p = start.next
                start.next = end
                while p != end:
                    tmp = p.next
                    del p
                    p = tmp
                p = start.next
                end = start
        return start_point.next

    # 83 Remove Duplicates from Sorted List, Easy
    def deleteDuplicates2(self, head: ListNode) -> ListNode:
        if head is None:
            return head

        pointer = head
        while pointer and pointer.next:
            if pointer.val == pointer.next.val:
                pointer.next = pointer.next.next
            else:
                pointer = pointer.next
        return head

    # 84 Largest Rectangle in Histogram, Hard
    def largestRectangleArea(self, heights: List[int]) -> int:
        # O(N^2) Time Limit Exceed..
        # size = len(heights)

        # if size == 1:
        #     return heights[0]

        # ans = 0
        # index = 0
        # while index < size:
        #     # 在局部峰值再进行判断，即只考虑大于后一个值的数所在位置的情况
        #     if index + 1 < size and heights[index] <= heights[index + 1]:
        #         index += 1
        #     else:
        #         minH = heights[index]
        #         for j in range(index, -1, -1):
        #             minH = min(minH, heights[j])
        #             area = minH * (index - j + 1)
        #             ans = max(ans, area)
        #         index += 1

        # return ans

        # O(N)
        res, stack = 0, []
        for j, x in enumerate(heights):
            i = j
            while stack and x <= stack[-1][1]:
                i, y = stack.pop()
                res = max(res, (j - i) * y)
            stack.append((i, x))
        while stack:
            i, y = stack.pop()
            res = max(res, (len(heights) - i) * y)
        return res

    # 85 Maximal Rectangle, Hard
    def maximalRectangle(self, matrix: List[List[str]]) -> int:
        # 这道题的解法灵感来自于Largest Rectangle in Histogram这道题，
        # 假设我们把矩阵沿着某一行切下来，然后把切的行作为底面，将自底面往上的矩阵看成一个直方图（histogram）。
        # 直方图的中每个项的高度就是从底面行开始往上1的数量。
        # 根据Largest Rectangle in Histogram中的largestRectangleArea函数我们就可以求出当前行作为矩阵下边缘的一个最大矩阵。
        # 接下来如果对每一行都做一次Largest Rectangle in Histogram，从其中选出最大的矩阵，那么它就是整个矩阵中面积最大的子矩阵。
        # 算法时间复杂度为O(m*n)，这解法真是太棒了，其实应该算是动态规划的题目了。
        if  matrix == [] or len(matrix) ==0:
            return 0
        maxArea = 0
        heights = [0 for i in range(len(matrix[0]))]
       
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                heights[j] = heights[j] + 1 if matrix[i][j] == '1' else 0 #计算高度值
            maxArea = max(maxArea,self.largestRectangleArea(heights))
        return maxArea   

    # 86 Partition List, Medium
    def partition(self, head: ListNode, x: int) -> ListNode:
        pre = ListNode(0)
        pre.next = head
        past = pre
        curr, end_list = head, head
        flag = False
        while curr:
            if curr.val >= x:
                flag = True
                end_list = curr
            else:
                if flag is False:
                    past = curr
                else:
                    tmp = past.next
                    past.next = curr
                    end_list.next = curr.next
                    curr.next = tmp

                    past = past.next
            curr = curr.next
    
        return pre.next

    # 88 Merge Sorted Array, Easy
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        if n == 0:
            return 
        j = 0
        i = 0
        size = m + n
        for index in range(size):
            if j >= n:
                break
            elif nums2[j] < nums1[index]:
                nums1[index + 1 : index + m - i + 1] = nums1[index : index + m - i]
                nums1[index] = nums2[j]
                j += 1
            elif index >= m + j:
                nums1[index] = nums2[j]
                j += 1
            else:
                i += 1

    # 89 Gray code, Medium
    def grayCode(self, n: int) -> List[int]:
        # 一个 n 位二进制的格雷码就是一个包含 2**n 种不同情况的列表，
        # 每一种情况的 n 位二进制数与其上一种情况的 n 位二进制数正好有一位不同
        # n 位二进制的格雷码生成方式如下：
        # 1.  n 位格雷码的 前 2**(n-1) 个代码字等于 n-1 位格雷码的代码字，按顺序书写，加前缀 0
        # 2. n 位格雷码的 后 2**(n-1) 个代码字等于 n-1 位格雷码的代码字，按逆序书写，加前缀 1
        # 1-bit: 0 1
        # 2-bit: 00 01 11 10 -> 0 + [0, 1] and 1 + [1, 0]
        nums = ['0', '1'] # 1-bit gray code

        for i in range(1, n):
            left = ['0' + num for num in nums]
            right = ['1' + num for num in nums[::-1]]
            nums = left + right
        return [int(num, 2) for num in nums]

    # 90 Subsets II, Medium
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        if len(nums) == 1:
            return [[], nums]
        res = []
        size = len(nums)
        def dfs(res, index, curr):
            curr.sort()
            if curr not in res:
                res.append(curr)

            for i in range(index, size):
                dfs(res, i + 1, curr + [nums[i]])
        
        dfs(res, 0, [])
        return res

            


def main():
    s = Solution()

    # 81
    # print(s.search(nums = [1,3,1,1,1], target = 3))
    # 82
    # nodes = connect_nodes([1, 5, 5, 5,5,5,5 ])
    # print_nodes(nodes)
    # print_nodes(s.deleteDuplicates(nodes))
    # 83
    # nodes = connect_nodes([1,2,3])
    # print_nodes(nodes)
    # print_nodes(s.deleteDuplicates2(nodes))
    # 84
    # print(len(heights))
    # heights = [2,3,5,6,2,3]
    # print(s.largestRectangleArea(heights))
    # 85
    # matrix =  [["1","0","1","0","1"],["0","1","0","1","0"],["1","0","1","0","1"],["0","1","0","1","0"]]
    # print(tabulate(matrix))
    # print(s.maximalRectangle(matrix))
    # 86
    # nodes = connect_nodes([2,1])
    # print_nodes(nodes)
    # print_nodes(s.partition(nodes, 2))
    # 88
    # nums1 =[4,5,6,0,0,0]
    # m = 3
    # nums2 = [1,2,3]
    # n = 3
    # print(nums1)
    # print(nums2)
    # s.merge(nums1, m, nums2, n)
    # print(nums1)
    # 89
    # print(s.grayCode(2))
    # 90
    print(s.subsetsWithDup([4,4,4,1,4]))


if __name__ == "__main__":
    main()
