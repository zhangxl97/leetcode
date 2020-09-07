from typing import List


class Solution:
    # 5479. Thousand Separator
    def thousandSeparator(self, n: int) -> str:
        s = list(str(n))
        n = len(s)
        i = n - 1
        while i > 2:
            s.insert(i - 2, '.')
            i -= 3
        return ''.join(s)

    # 5480. Minimum Number of Vertices to Reach All Nodes
    def get_val(self, x: List, i: int, has: list, full: List, dicts: dict):
        if i > len(x) - 1:
            return
        com = set(has + dicts[i])
        if com == set(full):
            # 符合条件记录
            x[i] = 1
            print(x)
            x[i] = 0
        x[i] = 1
        self.get_val(x, i + 1, list(com), full, dicts)
        # 当前位置不取 执行一次
        x[i] = 0
        self.get_val(x, i + 1, has, full, dicts)

    def findSmallestSetOfVertices(self, n: int, edges: List[List[int]]) -> List[int]:
        dicts = {i: [] for i in range(n)}
        for fr, to in edges:
            dicts[fr].append(to)

        full = [i for i in range(n)]
        temp = dicts[0]
        for i in range(1, n):
            temp += dicts[i]
        return list(set(full) - set(temp))

    # 5481. Minimum Numbers of Function Calls to Make Target Array
    # res = all ones in binary + (the largest number of bits - 1)
    def minOperations(self, nums: List[int]) -> int:
        return sum(bin(x).count('1') for x in nums) + len(bin(max(nums))) - 3   # bin(x) = 'obxxxxx', so -2 -1 = -3

def main():
    s = Solution()

    # 5479
    # print(s.thousandSeparator(123456712312312389))
    # 5480
    # print(s.findSmallestSetOfVertices(n = 5, edges = [[0,1],[2,1],[3,1],[1,4],[2,4]]))
    # 5481
    print(s.minOperations(nums = [4,2,5]))

if __name__ == '__main__':
    main()
