from typing import List

class Solution:
    # 69
    def mySqrt(self, x: int) -> int:
        if x == 0:
            return 0
        elif x <= 3:
            return 1

        left, right = 1, x // 2

        while left <= right:
            mid = left + (right - left) // 2
            tmp = mid * mid
            if tmp == x:
                return mid
            elif tmp > x:
                right = mid - 1
            else:
                left = mid + 1

        return right
    
    # 744
    def nextGreatestLetter(self, letters: List[str], target: str) -> str:
        
        left, right = 0, len(letters) - 1

        while left <= right:
            mid = left + (right - left) // 2

            if letters[mid] <= target:
                left = mid + 1
            else:
                right = mid - 1

        return letters[left] if left < len(letters) else letters[0]

    # 540, Easy
    def singleNonDuplicate(self, nums: List[int]) -> int:
        size = len(nums)
        left, right = 0, size - 1

        while left <= right:
            mid = left + (right - left) // 2

            if mid < size - 1 and nums[mid] == nums[mid + 1]:  # 和后面的数相等
                if (mid + 1) % 2 == 0:  # 单独的数出现在前面
                    right = mid - 1
                else:
                    left = mid + 2
                    

            elif mid > 0 and nums[mid] == nums[mid - 1]:  # 和前面的数相等
                if (mid + 1) % 2 == 1:  # 单独的数出现在前面
                    right = mid - 2
                else:  # 单独的数出现在后面
                    left = mid + 1
            else:
                return nums[mid]

    # 278, Easy
    def firstBadVersion(self, n):
        """
        :type n: int
        :rtype: int
        """

        left, right = 1, n
        
        while left < right:
            mid = (right + left) // 2
            if isBadVersion(mid):
                right = mid
            else:
                left = mid + 1

        return left

    # 153, Medium
    def findMin(self, nums: List[int]) -> int:

        left, right = 0, len(nums) - 1

        ans = nums[0]

        while left < right:
            mid = (right + left) // 2

            if nums[mid] <= nums[right]:
                right = mid
            else:
                left = mid + 1
        return nums[left]

    # 34, Medium
    def searchRange_(self, nums: List[int], target: int) -> List[int]:
        size = len(nums)
        if size == 0:
            return [-1, -1]

        left, right = 0, len(nums) - 1

        while left <= right:
            mid = (right + left) // 2
            if nums[mid] == target:
                mid_left, mid_right = mid, mid
                while mid_left > 0 and nums[mid_left - 1] == nums[mid]:
                    mid_left -= 1
                while mid_right < size - 1 and nums[mid_right + 1] == nums[mid]:
                    mid_right += 1
                return [mid_left, mid_right]
            elif nums[mid] > target:
                right = mid - 1
            else:
                left = mid + 1
        return [-1, -1]        
    
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        def binary_serach(nums, target):
            left, right = 0, len(nums) - 1
            while left <= right:
                mid = (right + left) // 2
                if nums[mid] >= target:
                    right = mid - 1
                else:
                    left = mid + 1
            return left
        
    
        first = binary_serach(nums, target)
        last = binary_serach(nums, target + 1) - 1

        if last < first:
            return [-1, -1]
        else:
            return [first, last]

        # 

    



def main():
    s = Solution()

    # print(s.mySqrt(2147395599))
    # print(s.nextGreatestLetter(letters = ["c", "f", "j"], target = "j"))
    # print(s.singleNonDuplicate([3,3,7,7,10,11,11]))
    # print(s.firstBadVersion(5))
    # print(s.findMin(nums = [2,3,4,5,1]))
    print(s.searchRange(nums = [0,1,1,3], target = 3))


if __name__ == "__main__":
    main()
