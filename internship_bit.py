from typing import List


class Solution:
    # 461. 汉明距离, Easy
    def hammingDistance(self, x: int, y: int) -> int:

        res = x ^ y
        cnt = 0
        while res:
            bit = res % 2
            res >>= 1
            if bit:
                cnt += 1
        return cnt
        
    # 136. 只出现一次的数字, Easy
    def singleNumber(self, nums: List[int]) -> int:
        if len(nums) == 1:
            return nums[0]
        for i in range(1, len(nums)):
            nums[i] = nums[i - 1] ^ nums[i]
        return nums[-1]
    
    # 268. 丢失的数字, Easy
    def missingNumber(self, nums: List[int]) -> int:
        return len(nums) * (len(nums) + 1) // 2 - sum(nums)
 
    # 260. 只出现一次的数字 III, Easy
    def singleNumber(self, nums: List[int]) -> List[int]:
        # from collections import Counter
        # nums = Counter(nums)
        # ans = []
        # for num in nums:
        #     if nums[num] == 1:
        #         ans.append(num)
        # return ans
        error = 0
        for num in nums:
            error ^= num
        error = error & (-error)
        res = [0,0]
        for num in nums:
            if num & error != 0:
                res[0] ^= num
            else:
                res[1] ^= num
        return res

    # 190. 颠倒二进制位, Easy
    def reverseBits(self, n: int) -> int:
        return int("0b" + bin(n)[2:].zfill(32)[::-1], base=2)

    # 231. 2的幂, Easy
    def isPowerOfTwo(self, n: int) -> bool:
        if n <= 0:
            return False
        if n == 1:
            return True

        while n > 1:
            n = n / 2
            if n % 1 != 0:
                return False
        return True
        # 1000 & 0111 = 0
        return n > 0 and (n & (n - 1)) == 0

    # 342. 4的幂, Easy
    def isPowerOfFour(self, n: int) -> bool:
        # if n <= 0:
        #     return False
        # if n == 1:
        #     return True
        
        # if n > 1:
        #     while n > 1:
        #         n = n / 4
        #         if n % 1 != 0:
        #             return False
        # else:
        #     while n < 1:
        #         n = n * 4
        # return n == 1
        

        return n > 0 and (n & (n - 1)) == 0 and (n & 0x55555555) != 0

    # 693. 交替位二进制数, Easy
    def hasAlternatingBits(self, n: int) -> bool:
        tmp = n ^ (n >> 1)
        return tmp & (tmp + 1) == 0

    # 476. 数字的补数, Easy
    def findComplement(self, num: int) -> int:
    
        # res = ""
        # while num > 0:
        #     if num % 2 == 0:
        #         res += "1"
        #     else:
        #         res += "0"
        #     num //= 2
        # res = res[::-1]
        
        # return int("0b"+res, base=2)
        mask = num
        mask |= mask >> 1
        mask |= mask >> 2
        mask |= mask >> 4
        mask |= mask >> 8
        mask |= mask >> 16
        return mask ^ num

    # 371. 两整数之和, Medium
    def getSum(self, a: int, b: int) -> int:
        # res = 0
        MASK = 0x100000000
        MAX_INT = 0x7FFFFFFF
        MIN_INT = MAX_INT + 1
        while b != 0:
            carry = (a & b) << 1
            a = (a ^ b) % MASK
            b = carry % MASK
        return a if a <= MAX_INT else ~((a % MIN_INT) ^ MAX_INT)

    # 318. 最大单词长度乘积, Medium
    def maxProduct(self, words: List[str]) -> int:
        # def check(s1, s2):
        #     for c in s1:
        #         index = s2.find(c)
        #         if index != -1:
        #             return False
        #     return True
        
        res = 0
        val = [0] * len(words)
        for i in range(len(words)):
            for c in words[i]:
                val[i] |= (1 << (ord(c) - 97))

        # s = []
        for i in range(len(words) - 1):
            for j in range(i + 1, len(words)):
                if val[i] & val[j] == 0:
                    res = max(res, len(words[i]) * len(words[j]))
                
        # print(s)
        return res

    # 338. 比特位计数, Medium
    def countBits(self, num: int) -> List[int]:
        if num == 0:
            return [0]
        elif num == 1:
            return [0, 1]

        dp = [0] * (num + 1)
        dp[1] = 1

        for i in range(2, num + 1):
            # 111000
            # dp[111000] = dp[11100] + 0
            dp[i] = dp[i // 2] + ((i % 2) == 1)

            # 111000
            # dp[111000] = dp[11] + dp[1000]
            # dp[i] = dp[i & (i - 1)] + 1
            # 前面的1 + 最后一位1
        
        return dp
    
    # 201. 数字范围按位与
    def rangeBitwiseAnd(self, left: int, right: int) -> int:
        # 如果有一位是0，区间内所有数字这一位相与（&）都为0。
        # k = 0
        # while left != right:
        #     left >>= 1
        #     right >>= 1
        #     k += 1
        # return left << k
        
        # 改进版，每次去掉最后一个1
        while left < right:
            right = right & (right - 1)
        return right

def main():
    s = Solution()

    # print(s.hammingDistance(1,4))
    # print(s.singleNumber([4,1,2,1,2]))
    # print(s.missingNumber([0]))
    # print(s.singleNumber(nums = [1,2,1,3,2,5]))
    # print(s.reverseBits(4294967293))
    # print(s.isPowerOfTwo(8))
    # print(s.isPowerOfFour(16))
    # print(s.hasAlternatingBits(3))
    # print(s.findComplement(10))
    # print(s.getSum(-1,1))
    # print(s.maxProduct( ["abcw","baz","foo","bar","xtfn","abcdef"]))
    print(s.countBits(2))

if __name__ == "__main__":
    main()
