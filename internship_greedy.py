from typing import List


class Solution:
    # 455. 分发饼干, Easy
    def findContentChildren(self, g: List[int], s: List[int]) -> int:
        g.sort()
        s.sort()
        i, j = 0, 0
        cnt = 0
        while i < len(g) and j < len(s):
            if g[i] > s[j]:
                j += 1
            else:
                cnt += 1
                i += 1
                j += 1

        return cnt

    # 435. 无重叠区间, Medium
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        nums = len(intervals)
        if nums <= 1:
            return 0
        intervals.sort()
        cnt = 0
        last_left, last_right = intervals[0]
        # print(last_left, last_right)

        for i in range(1, nums):
            left, right = intervals[i]
            if left >= last_right:
                last_left, last_right = left, right
            else:
                cnt += 1
                if last_right <= right:
                    continue
                else:
                    last_left, last_right = left, right
        return cnt
                
    # 452. 用最少数量的箭引爆气球, Medium
    def findMinArrowShots(self, points: List[List[int]]) -> int:
        num = len(points)
        if num == 0:
            return 0
        if num == 1:
            return 1

        points.sort()
        # intersections = [points[0]]
        intersections = points[0]
        index_inter = 0
        print(points)
        print(intersections)
        for i in range(1, num):
            left, right = points[i]
            # last_left, last_right = intersections[index_inter]
            last_left, last_right = intersections

            if left > last_right:
                # intersections.append([left, right])
                intersections = [left, right]
                index_inter += 1
            else:
                # intersections[index_inter] = [max(left, last_left), min(right, last_right)]
                intersections = [max(left, last_left), min(right, last_right)]
            # print(intersections)
        # return len(intersections)
        return index_inter + 1

    # 406. 根据身高重建队列, Medium ⭐
    # 身高 h 降序、个数 k 值升序，然后将某个学生插入队列的第 k 个位置中。
    def reconstructQueue(self, people: List[List[int]]) -> List[List[int]]:

        people=sorted(people,key=lambda x:(-x[0],x[1]))
        print(people)
        ans = []

        for p in people:
            if len(ans) <= p[1]:
                ans.append(p)
            else:
                ans.insert(p[1], p)
            print(ans)

        return ans

    # 121. 买卖股票的最佳时机, Easy
    def maxProfit_(self, prices: List[int]) -> int:
        max_profit = 0
        min_buy = prices[0]
        for price in prices[1:]:
            max_profit = max(max_profit, price - min_buy)
            min_buy = min(min_buy, price)
        return max_profit

    # 122. 买卖股票的最佳时机 II, Easy
    def maxProfit(self, prices: List[int]) -> int:
        if len(prices) == 1:
            return 0

        # buy = prices[0]
        # profit = 0
        # for i in range(1, len(prices)):
        #     if i < len(prices) - 1:
        #         if prices[i] > prices[i + 1]:
        #             profit += max(0, prices[i] - buy)
        #             buy = prices[i + 1]
        #         else:
        #             buy = min(buy, prices[i])
        #     else:
        #         profit += max(0, prices[i] - buy)
        # return profit

        profit = 0
        for i in range(1, len(prices)):
            if prices[i] > prices[i - 1]:
                profit += prices[i] - prices[i - 1]
        return profit

    # 605. 种花问题, Easy
    def canPlaceFlowers(self, flowerbed: List[int], n: int) -> bool:
        if n == 0:
            return True
        cnt = 0
        for i in range(len(flowerbed)):
            if flowerbed[i] == 1:
                continue
            else:
                pre = flowerbed[i - 1] if i > 0 else 0
                after = flowerbed[i + 1] if i < len(flowerbed) - 1 else 0
                if pre == 0 and after == 0:
                    flowerbed[i] = 1
                    cnt += 1
                    if cnt == n:
                        return True
        return False

    # 392. 判断子序列, Easy
    def isSubsequence(self, s: str, t: str) -> bool:
        # i = 0
        # j = 0
        # while i < len(s) and j < len(t):
        #     if s[i] == t[j]:
        #         i += 1
        #         j += 1
        #     else:
        #         j += 1
        # return i == len(s)
    
        index = -1
        for c in s:
            index = t.find(c, index + 1)
            if index == -1:
                return False
        return True

    # 665. 非递减数列, Easy
    def checkPossibility(self, nums: List[int]) -> bool:
        
        change = 0
        for i in range(1, len(nums)):
            if nums[i] >= nums[i - 1]:
                continue

            change += 1
            if change > 1:
                return False
            if i > 1 and nums[i - 2] > nums[i]:
                nums[i] = nums[i - 1]
            else:
                nums[i - 1] = nums[i]
        return True
 
    # 53. 最大子序和, Easy
    def maxSubArray(self, nums: List[int]) -> int:

        max_sum = nums[0]
        pre_sum = nums[0]
        for num in nums[1:]:
            pre_sum = pre_sum + num if pre_sum > 0 else num
            max_sum = max(max_sum, pre_sum)
        return max_sum

    # 763. 划分字母区间, Medium
    def partitionLabels(self, S: str) -> List[int]:

        positions = {}  # 每个字符出现的区间
        for i, c in enumerate(S):
            if positions.get(c) is None:
                positions[c] = [i, i]
            else:
                if positions[c][0] > i:
                    positions[c][0] = i
                else:
                    positions[c][1] < i
                    positions[c][1] = i
        
        ans = []
        past_left, past_right = positions[S[0]]
        index = past_right + 1
        # 找到区间交集的长度即可
        while index < len(S):
            c = S[index]
            left, right = positions[c]

            if left > past_right:
                ans.append(past_right - past_left + 1)
                past_left, past_right = left, right
            else:
                past_left = min(left, past_left)
                past_right = max(right, past_right)
            index += 1

        ans.append(past_right - past_left + 1)
        return ans


def main():
    s = Solution()

    # print(s.findContentChildren(g = [1,2], s = [1,2,3]))
    # print(s.eraseOverlapIntervals( [ [1,2], [2,3] ]))
    # print(s.findMinArrowShots(points = [[2,3],[2,3]]))
    print(s.reconstructQueue(people = [[7,0],[4,4],[7,1],[5,0],[6,1],[5,2]]))
    # print(s.maxProfit([7,6,4,3,1,2]))
    # print(s.maxProfit([7,1,5,3,6,4]))
    # print(s.canPlaceFlowers([0], 1))
    # print(s.isSubsequence("1", "4"))
    # print(s.checkPossibility([3,4,2,3]))
    # print(s.maxSubArray([1]))
    # print(s.partitionLabels(S = "ababcbacadefegdehijhklij"))

if __name__ == "__main__":
    main()
