from typing import List
from tabulate import tabulate


class Solution:
    # 41. First Missing Positive, Hard
    def firstMissingPositive(self, nums: List[int]) -> int:
        # size = len(nums)
        # if size == 0:
        #     return 1
        #
        # nums.sort()
        # if nums[0] > 1 or nums[-1] <= 0:
        #     return 1
        #
        # for i in range(1, size):
        #     if nums[i] <= 0:
        #         continue
        #     else:
        #         if nums[i - 1] > 0:
        #             if nums[i] == nums[i - 1] + 1 or nums[i] == nums[i - 1]:
        #                 continue
        #             else:
        #                 return nums[i - 1] + 1
        #         else:
        #             if nums[i] == 1:
        #                 continue
        #             else:
        #                 return 1
        #
        # return nums[-1] + 1
        if not nums:
            return 1
        pos = []
        for n in nums:
            if n > 0:
                pos.append(n)
        # nums = list(filter(lambda x: x > 0, nums))
        if not pos:
            return 1
        max_num = len(pos)
        pos = set(pos)

        for i in range(1, max_num + 2):
            if i not in pos:
                return i

    # 42. Trapping Rain Water, Hard
    def trap(self, height: List[int]) -> int:
        # size = len(height)
        # if size <= 2:
        #     return 0
        # res = 0
        # dp = [0] * size
        # dp[-1] = height[-1]
        # for i in range(size - 2, -1, -1):
        #     dp[i] = max(dp[i + 1], height[i])
        #
        # left_value = height[0]
        # for i in range(1, size):
        #     value = min(left_value, dp[i]) - height[i]
        #     if value > 0:
        #         res += value
        #     left_value = max(left_value, height[i])
        #
        #
        # return res

        if not height:
            return 0
        left, right = 0, len(height) - 1  # 左右指针
        area = 0
        leftwall, rightwall = 0, 0  # 左墙和右墙
        while (left < right):
            if height[left] < height[right]:  # 右边高，则以右端为墙
                if leftwall > height[left]:  # 如果左墙也比当前位置高的话
                    area += min(leftwall, height[right]) - height[left]  # 面积就是两墙最低者减去当前位置的高度
                else:
                    leftwall = height[left]  # 否则更新左墙
                left += 1
            else:
                if rightwall > height[right]:
                    area += min(rightwall, height[left]) - height[right]
                else:
                    rightwall = height[right]
                right -= 1
        return area

    # 43. Multiply Strings, Medium
    def multiply(self, num1: str, num2: str) -> str:
        return str(int(num1) * int(num2))

    # 44. Wildcard Matching, Hard
    def isMatch(self, s: str, p: str) -> bool:
        if p == s:
            return True
        elif p == "*":
            return True
        elif p == "" or s == "":
            return False

        # ues two pointer O(mn)
        # i = 0
        # size = len(p)
        # # 合并重复的*
        # while i < size - 1:
        #     if p[i] == "*" and p[i + 1] == "*":
        #         p = p[:i + 1] + p[i + 2:]
        #         size -= 1
        #     else:
        #         i += 1

        # i = 0
        # j = 0
        # p_star = -1
        # s_pos = -1
        # while i < len(s):

        #     if j < len(p) and (s[i] == p[j] or p[j] == '?'):
        #         i += 1
        #         j += 1
        #     elif j < len(p) and p[j] == "*":
        #         p_star = j
        #         j += 1
        #         s_pos = i
        #     else:
        #         if p_star >= 0:
        #             j = p_star + 1
        #             s_pos += 1
        #             i = s_pos
        #         else:
        #             return False


        # return (j == len(p)) or (j == len(p) - 1 and p[j] == "*")
        
        # Use DP
        m = len(p)
        n = len(s)
        T = [[None] * (n + 1) for _ in range(m + 1)]

        # Recursion Base Cases
        # (1) p: "" matches s: ""
        T[0][0] = True 

        # (2) p: "" does not match non-empty s
        for j in range(1, n + 1):
            T[0][j] = False

        # (3) For T[i][0] = True, p must be '*', '**', '***', etc. 
        #     Once p[i-1] != '*', all the T[i][0] afterwards will be False
        for i in range(1, m + 1):
            if p[i-1] == '*':
                T[i][0] = T[i-1][0]
            else:
                T[i][0] = False

        # Fill the table T (using recursion formulation -> see recursion code)      
        for i in range(1, m + 1):
            for j in range(1, n + 1):

                pIdx = i - 1  # Because p length in m and last index is m - 1
                sIdx = j - 1  # Because s length is n and last index is n - 1

                # (i-1)th char matches (j-1)th char or (i-1)th char matches single char
                if p[pIdx] == s[sIdx] or p[pIdx] == '?': 
                    T[i][j] = T[i-1][j-1]

                # (i-1)th char matches any sequence of chars
                elif p[pIdx] == '*':

                    # Recurrence relation in 2 variables: F(n,m) = F(n-1,m) + F(n,m-1)
                    T[i][j] = T[i-1][j] or T[i][j-1]

                else:
                    T[i][j] = False

        return T[m][n]

    # 45. Jump Game II, Hard
    def jump(self, nums: List[int]) -> int:
        # def check(curr, target, cnt, res):
        #     if curr > target or (curr < target and nums[curr] == 0):
        #         return res
        #     elif curr == target:
        #         if cnt < res:
        #             res = cnt
        #         return res

        #     for i in range(1, nums[curr] + 1):
        #         res = check(curr + i, target, cnt + 1, res)
        #     return res

        # # res = []
        # res = check(0, len(nums) - 1, 0, len(nums))
        # # print(res)
        # return res

        # Greedy 
        # We use "last" to keep track of the maximum distance that has been reached
        # by using the minimum steps "ret", whereas "curr" is the maximum distance
        # that can be reached by using "ret+1" steps. Thus,curr = max(i+A[i]) where 0 <= i <= last.
        # e.g. [2,3,1,1,4]
        ret = 0
        last = 0
        curr = 0
        for i in range(len(nums)):
            if i > last:
                last = curr
                ret += 1
                if last >= len(nums) - 1:
                    return ret
            curr = max(curr, i+nums[i])
        # return ret

    # 46. Permutations, Medium
    def permute(self, nums: List[int]) -> List[List[int]]:
        from itertools import permutations

        res = list(permutations(nums))
        return res
    
    # 47. Permutations II, Medium
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        # from itertools import permutations
        # res = list(set(list(permutations(nums))))
        # return res  
        
        def backtrack(nums, path, out):
            if not nums:
                out.append(path)
                return
              
                       
            for x in range(len(nums)): 
                if x > 0:
                    if nums[x - 1] == nums[x]:
                        continue
                
                backtrack(nums[:x] + nums[x + 1:], path + [nums[x]], out) 
        
        out = [] 
        nums.sort()
        backtrack(nums , [], out)
        
        return out

    # 48. Rotate Image, Medium
    def rotate(self, matrix: List[List[int]]) -> None:
        import numpy as np
        size = len(matrix)
        if size <= 1:
            return
        matrix.reverse()
        print(tabulate(matrix))
        for i in range(size):
            for j in range(i):
                matrix[i][j], matrix[j][i] = matrix[j][i],matrix[i][j]

    # 49. Group Anagrams, Medium
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        res = {}
        for word in strs:
            tmp = "".join(sorted(word))
            if res.get(tmp) is None:
                res[tmp] = [word]
            else:
                res[tmp].append(word)
        return list(res.values())
    
    # 50 Pow(x, n), Medium
    def myPow(self, x: float, n: int) -> float:
        # return pow(x, n)
        if n < 0:
            x = 1 / x
            n = -n
        elif n == 0:
            return 1
        elif n == 1:
            return x
        elif n == 2:
            return x * x
        elif n == 3:
            return x * x * x

        ans = 1
        base = x
        while (n):
            if n%2 == 1:
                ans = ans * base
            base = base *  base
            n = n // 2
        return ans


def main():
    s = Solution()

    # 41
    # print(s.firstMissingPositive([0,-1,3,1]))
    # 42
    # print(s.trap( [0,1,0,2,1,0,1,3,0,2,1,2,1]))
    # 43
    # print(s.multiply("2", "3"))
    # 44
    # print(s.isMatch(
    #     "abbabaaabbabbaababbabbbbbabbbabbbabaaaaababababbbabababaabbababaabbbbbbaaaabababbbaabbbbaabbbbababababbaabbaababaabbbababababbbbaaabbbbbabaaaabbababbbbaababaabbababbbbbababbbabaaaaaaaabbbbbaabaaababaaaabb",
    #     "**aa*****ba*a*bb**aa*ab****a*aaaaaa***a*aaaa**bbabb*b*b**aaaaaaaaa*a********ba*bbb***a*ba*bb*bb**a*b*bb"))
    # print(s.isMatch("aa", "*a"))
    # 45
    print(s.jump([2,3,1,1,4]))
    # 46
    # print(s.permute([1,2,3]))
    # 47
    # print(s.permuteUnique([1,1,3]))
    # 48
    # x = [[1,2],[3,4]]
    # print(tabulate(x))
    # s.rotate(x)
    # print(tabulate(x))
    # 49
    # print(s.groupAnagrams(["eat","tea","tan","ate","nat","bat","ac","bd","aac","bbd","aacc","bbdd","acc","bdd"]))
    # 50
    # print(s.myPow(2.00000, 5))

if __name__ == '__main__':
    main()
