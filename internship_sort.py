from typing import List


class Solution:
    # 215. 数组中的第K个最大元素, Medium
    def findKthLargest(self, nums: List[int], k: int) -> int:
        # nums.sort(reverse=True)
        # return nums[k - 1]

        # def quick_sort(nums, left, right):
        #     if left < right:

        #         i, j = left, right
        #         pivot = nums[left]
        #         while i != j:

        #             while j > i and nums[j] > pivot:
        #                 j -= 1
        #             if j > i:
        #                 nums[i] = nums[j]
        #                 i += 1
                    
        #             while i < j and nums[i] < pivot:
        #                 i += 1
        #             if i < j:
        #                 nums[j] = nums[i]
        #                 j -= 1
                    
        #         nums[i] = pivot

        #         quick_sort(nums, left, i - 1)
        #         quick_sort(nums, i + 1, right)
        
        # quick_sort(nums, 0, len(nums) - 1)
        # # print(nums)
        # return nums[len(nums) - k]

        # heapq 最大堆
        import heapq

        heap = []
        for n in nums:
            heapq.heappush(heap, n)
            if len(heap) > k:
                heapq.heappop(heap)

        return heapq.heappop(heap)

    # 347. 前 K 个高频元素, Medium
    # 桶排序, 频率
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        from collections import Counter
        size = len(nums)
        buckets = {i:[] for i in range(1, size + 1)}
        nums = Counter(nums)

        # for num in nums:
        #     buckets[nums[num]].append(num)
        
        # ans = []
        # ans_size = 0
        # for i in range(size, 0, -1):
        #     tmp_size = len(buckets[i])
        #     if tmp_size != 0:
        #         ans_size += tmp_size
        #         ans += buckets[i]
        #         if ans_size == k:
        #             break
        # return ans

        import heapq
        heap = []

        for key, value in nums.items():
            # 按-value的升序排列，栈顶为-value最小的值，即value（出现次数最多）的值
            heapq.heappush(heap, (-value, key))
        
        ans = []
        for _ in range(k):
            ans.append(heapq.heappop(heap)[1])
        return ans

    # 451. 根据字符出现频率排序, Medium
    def frequencySort(self, s: str) -> str:
        from collections import Counter
        import heapq

        s = Counter(s)

        heap = []
        for key, value in s.items():
            # 需要最大堆，就(value, key)  --> 输出时从小到大
            # 需要最小堆，就(-value, key)  --> 输出时从大(value取负)到小
            heapq.heappush(heap, (-value, key))

        ans = ""
        while heap: 
            value, key = heapq.heappop(heap)
            ans += key * (-value)
        return ans

    # 75. 颜色分类, Medium
    def sortColors(self, nums: List[int]) -> None:
        zero, one, two = -1, 0, len(nums)
        while one < two:
            if nums[one] == 0:
                zero += 1
                nums[zero], nums[one] = nums[one], nums[zero]
                one += 1
            elif nums[one] == 2:
                two -= 1
                nums[one], nums[two] = nums[two], nums[one]
            else:
                one += 1

def main():
    s = Solution()

    # nums = [3,2,3,1,2,4,5,5,6]
    # print(s.findKthLargest(nums, 4))
    
    # print(s.topKFrequent(nums = [1,1,1,2,2,3], k = 2))

    # print(s.frequencySort("tree"))

    print(s.sortColors(nums = [2,0,1]))

if __name__ == "__main__":
    main()
