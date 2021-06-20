def binary_search(nums, target):
    left, right = 0, len(nums) - 1

    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1


# print(binary_search([1,2,3,4,5,6,7],8))
 

def binary_search_2(nums, target):
    left, right = 0, len(nums) - 1

    while left < right:
        mid = (left + right) // 2
        if nums[mid] < target:
            left = mid + 1
        else:
            right = mid
    
    return left

# from bisect import bisect, bisect_left, bisect_right
# print(binary_search_2([0,1,1,1,3,3,3],3))
# print(bisect([0,1,1,1,3,3,3],3))
# print(bisect_left([0,1,1,1,3,3,3],3))
# print(bisect_right([0,1,1,1,3,3,3],3))


def max_range(nums):

    pre_max = nums[0]
    ans = 0

    for num in nums[1:]:
        pre_max = max(pre_max + num, num)
        ans = max(ans, pre_max)
        
    return ans

print(max_range([-2, 0, 3, 5, -20, 100]))