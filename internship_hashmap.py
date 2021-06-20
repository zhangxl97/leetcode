from typing import List

class Solution:
    # 1
    def twoSum(self, nums: List[int], target: int) -> List[int]:

        kvs = {}
        for i, n in enumerate(nums):
            if kvs.get(n) is None:
                kvs[n] = [i]
            else:
                kvs[n].append(i)
        
        for i, n in enumerate(nums):
            rest = target - n
            if rest == n:
                if len(kvs[rest]) == 2:
                    return kvs[rest]
            else:
                if kvs.get(rest):
                    return [i, kvs[rest][0]]
        
        return False

    # 594, Easy
    def findLHS(self, nums: List[int]) -> int:
        from collections import Counter
        nums = Counter(nums)
        ans = 0
        for num in nums:
            if nums[num + 1] != 0:
                tmp = nums[num] + nums[num + 1]
                if tmp > ans:
                    ans = tmp
        return ans

    # 128, Hard
    def longestConsecutive(self, nums: List[int]) -> int:
        from collections import Counter

        nums = Counter(nums)
        visited = {num: False for num in nums}

        max_len = 0
        for num in nums:
            if visited[num] is False:
                cnt = 1
                visited[num] = True
                tmp = num
                while nums[tmp - 1] != 0 and visited[tmp - 1] is False:
                    cnt += 1
                    tmp -= 1
                    visited[tmp] = True
                tmp = num
                while nums[tmp + 1] != 0 and visited[tmp + 1] is False:
                    cnt += 1
                    tmp += 1
                    visited[tmp] = True
                max_len = max(max_len, cnt)
        return max_len


def main():
    s = Solution()

    # print(s.twoSum(nums = [1,3,3], target = 6))
    # print(s.findLHS(nums = [1,1,1,1]))
    print(s.longestConsecutive(nums = [0,3,7,2,5,8,4,6,0,1]))


if __name__ == "__main__":
    main()
                    

