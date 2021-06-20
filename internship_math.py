from typing import List

class Solution:
    # 204. 计数质数, 204
    # 埃拉托斯特尼筛法
    def countPrimes(self, n: int) -> int:
        if n <= 2:
            return 0

        is_prime = [1] * (n)
        is_prime[0] = 0
        is_prime[1] = 0
        # # 去除所有偶数
        # for i in range(2, n):
        #     if i > 2 and i % 2 == 0:
        #         is_prime[i] = 0

        # i = 3
        # while i * i <= n:
        #     if is_prime[i] == 1:
        #     # 去除i的倍数
        #         j = i + i
        #         while j < n:
        #             is_prime[j] = 0
        #             j += i

        #     i += 2
            
        # return sum(is_prime)

        for i in range(2, int(n ** 0.5) + 1):
            if is_prime[i]:
                is_prime[i * i : n : i] = [0] * ((n - 1 - i * i) // i + 1)
        return sum(is_prime)

    # gcd 最大公约数
    # greatest common divisor
    # lcm 最小公倍数
    # Least Common Multiple
    def gcd(self, a, b):
        # 辗转相除
        while b:
            tmp = a % b
            a = b
            b = tmp
        
        return a
    def lcm(self, a, b):
        return a * b / self.gcd(a, b)

    # 504. 七进制数, Easy
    def convertToBase7(self, num: int) -> str:
        if num == 0:
            return "0"

        ans = ""
        flag = 0
        if num < 0:
            flag = 1
            num = -num
        
        while num:
            ans += str(num % 7)
            num //= 7
        return ans[::-1] if flag == 0 else "-" + ans[::-1]

    # 405. 数字转换为十六进制数, Easy
    def toHex(self, num: int) -> str:
        if num == 0:
            return "0"
        table = {0:'0', 1:'1', 2:'2', 3:'3', 4:'4', 5:'5', 6:'6', 7:'7', 8:'8', 9:'9', 10:'a', 11:'b', 12:'c', 13:'d', 14:'e', 15:'f'}

        if num < 0:
            num = 2 ** 32 - 1 + num + 1

        ans = ""
        
        while num:
            ans += table[num % 16]
            num //= 16
        
        return ans[::-1]

    # 168. Excel表列名称, Easy
    def convertToTitle(self, columnNumber: int) -> str:
        table = {i:chr(65 + i) for i in range(26)}
        
        ans = ""
        while columnNumber:
            columnNumber -= 1
            ans += table[columnNumber % 26]
            columnNumber //= 26

        return ans[::-1]

    # 172. 阶乘后的零, Easy 看能分解出多少个2*5， 2的数量明显多于5，所以仅需统计5的数量
    def trailingZeroes(self, n: int) -> int:
        if n < 4:
            return 0

        fives = 0
        while n:
            fives += n // 5
            n //= 5
        return fives

    # 67. 二进制求和, Easy
    def addBinary(self, a: str, b: str) -> str:
        return bin(int(a,2) + int(b,2))[2:]

    # 415. 字符串相加, Easy
    def addStrings(self, num1: str, num2: str) -> str:
        num1 = num1[::-1]
        num2 = num2[::-1]
        ans = ""
        carry = 0
        size1, size2 = len(num1), len(num2)
        i1, i2 = 0, 0
        while i1 < size1 or i2 < size2:
            if i1 == size1:
                tmp = int(num2[i2]) + carry
                carry = tmp // 10
                tmp = tmp % 10
                ans += str(tmp)
                i2 += 1
            elif i2 == size2:
                tmp = int(num1[i1]) + carry
                carry = tmp // 10
                tmp = tmp % 10
                ans += str(tmp)
                i1 += 1
            else:
                tmp = int(num1[i1]) + int(num2[i2]) + carry
                carry = tmp // 10
                tmp = tmp % 10
                ans += str(tmp)
                i1 += 1
                i2 += 1
            
        if carry:
            ans += str(carry)

        return ans[::-1]

    # 462. 最少移动次数使数组元素相等 II, Medium
    def minMoves2(self, nums: List[int]) -> int:
        nums.sort()
        left, right = 0, len(nums) - 1
        ans = 0
        while left <= right:
            ans += nums[right] - nums[left]
            left += 1
            right -= 1
        return ans

    # 169. 多数元素, Easy
    def majorityElement(self, nums: List[int]) -> int:
        # from collections import Counter
        # size = len(nums)
        # nums = Counter(nums)
        # for num, cnt in nums.items():
        #     if cnt > size // 2:
        #         return num
        
        # Boyer-Moore Majority Vote Algorithm 
        cnt = 0
        major = nums[0]
        for num in nums:
            major = num if cnt == 0 else major
            cnt = cnt + 1 if major == num else cnt - 1
        return major

    # 367. 有效的完全平方数, Easy
    # 平方数序列差值为等差数列：1 4 9 16 --> 3 5 7
    def isPerfectSquare(self, num: int) -> bool:
        inter = 1
        while num > 0:
            num -= inter
            inter += 2
        return num == 0

    # 326. 3的幂, Easy
    def isPowerOfThree(self, n: int) -> bool:
        if n <= 0:
            return False
            
        while n > 1:
            n = n / 3
            deci = n % 1
            if deci != 0:
                return False
        return True

    # 238. 除自身以外数组的乘积, Medium
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        size = len(nums)
        ans = [1] * size

        left = nums[0]
        for i in range(1, size):
            ans[i] *= left
            left *= nums[i]
        right = nums[size - 1]
        for i in range(size - 2, -1, -1):
            ans[i] *= right
            right *= nums[i]
        return ans

    # 628. 三个数的最大乘积, Easy
    def maximumProduct(self, nums: List[int]) -> int:
        max1, max2, max3 = -float('inf'), -float('inf'), -float('inf')
        min1, min2 = float('inf'), float('inf')

        for num in nums:
            if num > max1:
                max3 = max2
                max2 = max1
                max1 = num
            elif num > max2:
                max3 = max2
                max2 = num
            elif num > max3:
                max3 = num
            
            if num < min1:
                min2 = min1
                min1 = num
            elif num < min2:
                min2 = num
        return max(max1 * max2 * max3, max1 * min1 * min2)


def main():
    s = Solution()

    # print(s.countPrimes(10))
    # print(s.gcd(25, 10))
    # print(s.lcm(25, 10))

    # print(s.convertToBase7(8))
    # print(s.toHex(-1))
    # print(s.convertToTitle(2))
    # print(s.trailingZeroes(10))
    # print(s.addBinary("1010", "1011"))
    # print(s.addStrings(num1 = "9", num2 = "1"))
    # print(s.minMoves2([1,2,3]))
    # print(s.majorityElement([2,2,1,1,1,2,2]))
    # print(s.isPerfectSquare(16))
    # print(s.isPowerOfThree(45))
    # print(s.productExceptSelf([1,2,3,4]))
    print(s.maximumProduct([1,2,3,4, -5,-6]))


if __name__ == "__main__":
    main()
