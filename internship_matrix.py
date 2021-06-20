from typing import List
from tabulate import tabulate


class Solution:
    # 283. 移动零, Easy
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        # left, right = 0, len(nums) - 1

        # while right > left and nums[right] == 0:
        #     right -= 1
        # while left < right and nums[left] != 0:
        #     left += 1

        # while left <= right:
        #     if nums[left] != 0:
        #         left += 1
        #     else:
        #         nums[left:right] = nums[left+1:right+1]
        #         nums[right] = 0
        #         right -= 1
        idx = 0
        for num in nums:
            if num != 0:
                nums[idx] = num
                idx += 1
        
        for i in range(idx, len(nums)):
            nums[i] = 0
    
    # 566. 重塑矩阵, Easy
    def matrixReshape(self, nums: List[List[int]], r: int, c: int) -> List[List[int]]:
        rows = len(nums)
        cols = len(nums[0])
        if rows * cols != r * c:
            return nums
        
        ans = [[0 for _ in range(c)] for _ in range(r)]
        row_pre = 0
        col_pre = 0

        for row in range(r):
            for col in range(c):
                ans[row][col] = nums[row_pre][col_pre]
                col_pre += 1
                if col_pre == cols:
                    col_pre = 0
                    row_pre += 1
        return ans

    # 485. 最大连续 1 的个数, Easy
    def findMaxConsecutiveOnes(self, nums: List[int]) -> int:
        res = 0
        local = 0
        for num in nums:
            if num == 0:
                if local > res:
                    res = local
                local = 0
            else:
                local += 1
        if local > res:
            res = local
        return res

    # 240. 搜索二维矩阵 II, Medium
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        rows = len(matrix)
        cols = len(matrix[0])

        row, col = 0, cols - 1
        while row < rows and col >= 0:
            if matrix[row][col] == target:
                return True
            elif matrix[row][col] > target:
                col -= 1
            else:
                row += 1

        return False
                



def main():
    s = Solution()

    # array = [0,1,0,2,3]
    # print(array)
    # s.moveZeroes(array)
    # print(array)
    # print(s.matrixReshape(nums = [[1,2], [3,4]],r = 2, c = 4))
    # print(s.findMaxConsecutiveOnes([1,1,0,1,1,1]))
    matrix = [[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]]
    print(tabulate(matrix))
    print(s.searchMatrix(matrix, target = 20))


if __name__ == "__main__":
    main()
