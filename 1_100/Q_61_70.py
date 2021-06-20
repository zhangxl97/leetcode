from typing import List


# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    # 61 Rotate list, Medium
    def rotateRight(self, head: ListNode, k: int) -> ListNode:
        if head is None:
            return None
        size = 0
        p = head
        while p:
            size += 1
            p = p.next
        k = k % size
        if k == 0:
            return head
        tmp = None
        p = head
        for _ in range(size - k - 1):
            p = p.next
        tmp = p.next
        p.next = None
        p = tmp
        while tmp.next:
            tmp = tmp.next
        tmp.next = head

        return p

    # 62 Unique Paths, Medium
    def uniquePaths(self, m: int, n: int) -> int:
        if m == 1 or n == 1:
            return 1
        # DFS time limited
        # self.cnt = 0
        # def search(x, y):
        #     if x == m and y == n:
        #         self.cnt += 1
        #         return 
        #     if x < m:
        #         search(x + 1, y)
        #     if y < n:
        #         search(x, y + 1)
        # search(1, 1)
        # return self.cnt
        dp = [[1 for _ in range(n)] for _ in range(m)]
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
        return dp[m - 1][n - 1]

    # 63 Unique Paths II, Medium
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        m = len(obstacleGrid)
        n = len(obstacleGrid[0])

        dp = [[0 for _ in range(n)] for _ in range(m)]

        if obstacleGrid[0][0] == 0:
            dp[0][0] = 1
        for i in range(1, m):
            dp[i][0] = 1 if obstacleGrid[i][0] == 0 and dp[i - 1][0] == 1 else 0
        for j in range(1, n):
            dp[0][j] = 1 if obstacleGrid[0][j] == 0 and dp[0][j - 1] == 1 else 0

        for i in range(1, m):
            for j in range(1, n):
                if obstacleGrid[i][j] == 0:
                    dp[i][j] = dp[i - 1][j] + dp[i][j - 1]

        return dp[-1][-1]

    # 64. Minimum Path Sum, Medium
    def minPathSum(self, grid: List[List[int]]) -> int:
        m = len(grid)
        n = len(grid[0])

        dp = [[0 for _ in range(n)] for _ in range(m)]
        dp[0][0] = grid[0][0]
        for i in range(1, m):
            dp[i][0] = grid[i][0] + dp[i - 1][0]
        for j in range(1, n):
            dp[0][j] = grid[0][j] + dp[0][j - 1]

        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j]

        return dp[-1][-1]

    # 65 Valid Number, Hard
    def isNumber(self, s: str) -> bool:
        try:
            val = float(s)
            return True
        except:
            return False
        # print(val)

    # 66 Plus One, Easy
    def plusOne(self, digits: List[int]) -> List[int]:
        # slow faster than 13%
        # size = len(digits)
        # digits = list(map(str, digits))
        # val = str(int(''.join(digits)) + 1)
        # val = val.zfill(size)
        # # print(val)
        # return list(map(int, val))

        size = len(digits)
        digits[-1] += 1
        carry = digits[-1] // 10
        for i in range(size - 1, -1, -1):
            if carry == 0:
                return digits
            else:
                digits[i] = 0
                if i > 0:
                    digits[i - 1] += 1
                    carry = digits[i - 1] // 10
                else:
                    digits.insert(0, 1)
                    return digits

    # 67 Add Binary, Easy
    def addBinary(self, a: str, b: str) -> str:
        return bin(int(a, base=2) + int(b, base=2))[2:]

    # 68 Text Justification, Hard
    def fullJustify(self, words: List[str], maxWidth: int) -> List[str]:
        print(words)
        size = len(words)
        lengths = [len(words[i]) for i in range(size)]
        if size == 1:
            return [words[0].ljust(maxWidth)]
        index = 0
        res = []
        while index < size:
            word = ""
            num_of_word = 0
            if index == size - 1:
                # word = words[-1]#.ljust(maxWidth)
                num_of_word = 1
                index += 1
            else:
                length_tmp = len(words[index])
                length = length_tmp
                # word += words[index]
                num_of_word += 1
                while index < size:
                    index += 1
                    if index == size:
                        # word = word.ljust(maxWidth)
                        break

                    length_tmp = len(words[index])
                    length = length + 1 + length_tmp
                    if length <= maxWidth:
                        # word += " " + words[index]
                        num_of_word += 1
                    else:
                        break
            # word = word.ljust(maxWidth)
            if index == size:
                word = words[index - num_of_word]
                for i in range(index - num_of_word + 1, index):
                    word += " " + words[i]
                word = word.ljust(maxWidth)
            else:
                length_of_word = sum(lengths[index - num_of_word: index])
                length_of_blank = maxWidth - length_of_word
                if num_of_word == 1:
                    word = words[index - num_of_word].ljust(maxWidth)
                # elif length_of_blank % (num_of_word - 1) == 0:
                #     each_blank = length_of_blank // (num_of_word - 1)
                #     word = words[index - num_of_word]
                #     for i in range(index - num_of_word + 1, index):
                #         word += " " * each_blank + words[i]
                else:
                    word = words[index - num_of_word]
                    for i in range(index - num_of_word + 1, index):
                        if length_of_blank % (num_of_word - 1) == 0:
                            blank = length_of_blank // (num_of_word - 1)
                            length_of_blank -= blank
                            num_of_word -= 1
                        else:
                            blank = length_of_blank // (num_of_word - 1) + 1
                            length_of_blank -= blank
                            num_of_word -= 1
                        word += " " * blank + words[i]

            res.append(word)
            # print(word, len(word))
            # print(num_of_word, index, length_of_word, length_of_blank)

            # res.append(word)

        return res

    # 69 Sqrt(x), Easy
    def mySqrt(self, x: int) -> int:
        if x <= 2:
            return 1
        
        left = 0
        right = x
        while left < right:
            mid = left + (right - left) // 2
            if mid ** 2 > x:
                right = mid 
            elif mid ** 2 == x:
                return mid
            else:
                left = mid + 1
        
        return right - 1

    # 70 Climbing Stairs, Easy
    def climbStairs(self, n: int) -> int:
        if n <= 2:
            return n
        
        dp = [0 for _ in range(n)]
        dp[0] = 1
        dp[1] = 2
        for i in range(2, n):
            dp[i] = dp[i - 1] + dp[i - 2]
        return dp[-1]

def generate(nums: List) -> ListNode:
    if nums == []:
        return None

    head = ListNode(0)
    p = head
    for n in nums:
        p.next = ListNode(n)
        p = p.next
    return head.next


def print_nodes(head: ListNode) -> None:
    p = head
    while p:
        print(p.val, end="")
        if p.next:
            print("->", end="")
        p = p.next
    print("\r\n")


def main():
    s = Solution()

    # 61
    # l1 = generate([])
    # print_nodes(l1)
    # print_nodes(s.rotateRight(l1, 4))
    # 62
    # print(s.uniquePaths(7, 3))
    # 63 
    # print(s.uniquePathsWithObstacles( [[0,0,0],[0,1,0],[0,0,0]]))
    # 64
    # print(s.minPathSum(grid = [[1],[1],[4]]))
    # 65
    # print(s.isNumber("95a54e53"))
    # 66
    # print(s.plusOne(digits = [4,3,2,1]))
    # 67
    # print(s.addBinary(a = "1010", b = "1011"))
    # print(s.fullJustify(words = ["Science","is","what","we","understand","well","enough","to","explain","to","a","computer.","Art","is","everything","else","we","do"], maxWidth = 20))
    # 69
    # print(s.mySqrt(5))
    # 70
    print(s.climbStairs(45))

if __name__ == "__main__":
    main()
