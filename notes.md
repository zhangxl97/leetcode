# Notes

## Index

> Github: https://github.com/zhangxl97/leetcode.git

1. [Number Problems](#jump1)
2. [Linked list](#jump2)
3. [String](#jump3)
4. [Sort](#jump4)
5. [Stack](#jump5)
6. [DFS](#jump6)
7. [DP](#jump7)

## **<span id="jump1">1 Number Problems</span>**

- No.1  Two Sum, Easy

  > confirm whether target - n is in the rest list
  >
  > use dictionary

- No.9  Palindrome Number, Easy

  > str(x) == str(x)[::-1]

- No.11 Container With Most Water, Medium

  >use ***two pointer*** to indicate the two lines
  >
  >compare the height of these two lines --> left is lower, move left, if not, move right

- No.12 Integer to Roman, Medium and No.13 Roman to Integer,  Easy

  > compare with each base

- No.15 3Sum, Medium

  > Method one: O(n^2)   ***two pointer***
  >
  > ```python
  > nums.sort()
  >         res = []
  >         i = 0
  >         while i < size - 2:
  >             t = target - nums[i]
  >             left = i + 1
  >             right = size - 1
  >             while left < right:
  >                 if nums[left] + nums[right] == t:
  >                     res.append([nums[i], nums[left], nums[right]])
  >                     while nums[left] == nums[left + 1] and left < right - 1:
  >                         left += 1
  >                     left += 1
  >                     while nums[right] == nums[right - 1] and left < right - 1:
  >                         right -= 1
  >                     right -= 1
  >                 elif nums[left] + nums[right] < t:
  >                     left += 1
  >                 else:
  >                     right -= 1
  >             while nums[i] == nums[i + 1] and i < size - 3:
  >                 i += 1
  >             i += 1
  > ```
  >
  > Method two:  O(nlogn) use package ***bisect*** ?
  >
  > ```python
  > nums.sort()
  > count = Counter(nums)
  > keys = list(count.keys())
  > keys.sort()
  > if target / 3 in keys and count[target / 3] >= 3:
  >     res.append([target//3] * 3)
  > 
  > begin = bisect_left(keys, target - keys[-1] * 2)
  > end = bisect_left(keys, target * 3)
  > for i in range(begin, end):
  >     a = keys[i]
  >     if count[a] >= 2 and target - 2 * a in count:
  >         res.append([a, a, target - 2 * a])
  > 
  >         max_b = (target - a) // 2  # target-a is remaining
  >         min_b = target - a - keys[-1]  # target-a is remaining and c can max be keys[-1]
  >         b_begin = max(i + 1, bisect_left(keys, min_b))
  >         b_end = bisect_right(keys, max_b)
  > 
  >         for j in range(b_begin, b_end):
  >             b = keys[j]
  >             c = target - a - b
  >             if c in count and b <= c:
  >                 if b < c or count[b] >= 2:
  >                     res.append([a, b, c])
  > ```
  
- No.16 3Sum Closest, Medium

  > use ***two pointer***

- No.18 4Sum

  > use two pointer --> O(n^3)
  >
  > But one amazing algorithm !!!
  >
  > ```python
  >         n = nums
  >         t = target
  >         n.sort()
  >         if not n:
  >             return []
  >         L, N, S, M = len(n), {j: i for i, j in enumerate(n)}, [], n[-1]
  >         for i in range(L - 3):
  >             a = n[i]
  >             if a + 3 * M < t:
  >                 continue
  >             if 4 * a > t:
  >                 break
  >             for j in range(i + 1, L - 2):
  >                 b = n[j]
  >                 if a + b + 2 * M < t:
  >                     continue
  >                 if a + 3 * b > t:
  >                     break
  >                 for k in range(j + 1, L - 1):
  >                     c = n[k]
  >                     d = t - (a + b + c)
  >                     if d > M:
  >                         continue
  >                     if d < c:
  >                         break
  >                     if d in N and N[d] > k and [a, b, c, d] not in S:
  >                         S.append([a, b, c, d])
  >         return S
  > ```
  >

- No.26 Remove Duplicates from Sorted Array, Easy

  > Remove duplicates "in-place" It's easy By Python, we can use set().

- No.27 Remove Element, Easy

  > "in-place" --> nums.count(val) and nums.remove(val)

- No.28 Implement strStr(), Easy

  > str.find() --> KMP algorithm, BM algorithm, Sunday...

- No.29 Divide Two Integers, Medium

- No.31 Next Permutation

  > [hint](https://blog.csdn.net/Dby_freedom/article/details/85226270)
  >
  > ```python
  >         left_i = len(nums) - 2
  >         right_i = len(nums) - 1
  > 
  >         # e.g 2 3 1
  >         # left_i: 1, right_i: 2
  >         while left_i >= 0 and nums[left_i] >= nums[left_i + 1]:
  >             left_i -= 1
  >         # left_i: 0
  >         if left_i >= 0:
  >             # 2 3 1
  >             while nums[right_i] <= nums[left_i]:
  >                 right_i -= 1
  >             nums[left_i], nums[right_i] = nums[right_i], nums[left_i]
  >         nums[left_i + 1:] = sorted(nums[left_i + 1:])
  > ```

- No.36 Valid Sudoku, Medium

  > save each number's position, then use set

- No.38 Count and Say, Easy

- No.41 First Missing Positive, Hard

  > ```python
  > if not nums:
  >     return 1
  > nums = list(filter(lambda x: x > 0, nums))  # all positive numbers
  > if not nums:
  >     return 1
  > max_num = len(nums)
  > nums = set(nums)
  > for i in range(1, max_num + 2):
  >  if i not in nums:
  >         return i
  >    ```

- No.42 Trapping Rain Water, Hard

  > ```python
  > # min(max left, max right) - hight[i]
  > size = len(height)
  > if size <= 2:
  >     return 0
  > res = 0
  > dp = [0] * size
  > dp[-1] = height[-1]
  > for i in range(size - 2, -1, -1):
  >     dp[i] = max(dp[i + 1], height[i])
  > 
  > left_value = height[0]
  > for i in range(1, size):
  >     value = min(left_value, dp[i]) - height[i]
  >     if value > 0:
  >         res += value
  >     left_value = max(left_value, height[i])
  > return res
  > ```
  >
  > ```python
  > # two pointer
  > if not height: 
  >     return 0
  > left, right = 0 , len(height)-1         # 左右指针
  > area = 0 
  > leftwall, rightwall = 0,0               # 左墙和右墙
  > while(left<right):
  > 	if height[left]<height[right]:      # 右边高，则以右端为墙
  > 	    if leftwall>height[left]:       # 如果左墙也比当前位置高的话  
  > 		    area+=min(leftwall,height[right])-height[left]       # 面积就是两墙最低者减去当前位置的高度
  > 	    else:
  >             leftwall = height[left]     # 否则更新左墙
  >         left+=1                  
  >     else:
  >         if rightwall>height[right]:
  >             area+=min(rightwall,height[left])-height[right]
  >         else:
  >             rightwall = height[right]
  >         right-=1
  > return area
  > ```

- Np.45 Jump Game II, Hard

  > Method one: use DFS  --> Time Limited
  >
  > ```python
  >         def check(curr, target, cnt, res):
  >             if curr > target or (curr < target and nums[curr] == 0):
  >                 return res
  >             elif curr == target:
  >                 if cnt < res:
  >                     res = cnt
  >                 return res
  > 
  >             for i in range(1, nums[curr] + 1):
  >                 res = check(curr + i, target, cnt + 1, res)
  >             return res
  > 
  >         # res = []
  >         res = check(0, len(nums) - 1, 0, len(nums))
  >         # print(res)
  >         return res
  > ```
  >
  > Method Two:
  >
  > ​    \# Greedy 
  >
  > ​    \# We use "last" to keep track of the maximum distance that has been reached
  >
  > ​    \# by using the minimum steps "ret", whereas "curr" is the maximum distance
  >
  > ​    \# that can be reached by using "ret+1" steps Thus,curr = max(i+A[i]) where 0 <= i <= last.
  >
  > ```python
  >         ret = 0
  >         last = 0
  >         curr = 0
  >         for i in range(len(nums)):
  >             if i > last:
  >                 last = curr
  >                 ret += 1
  >             curr = max(curr, i+nums[i])
  >         return ret
  > ```

- No.46 Permutations, Medium

  > itertools.permutations

- No.48 Rotate Image, Medium

  > ```python
  >         size = len(matrix)
  >         if size <= 1:
  >             return
  >         matrix.reverse()
  >         for i in range(size):
  >             for j in range(i):
  >                 matrix[i][j], matrix[j][i] = matrix[j][i],matrix[i][j]
  > ```

- No.49 Group Anagrams, Medium

  > sorted word --> key

- No.50 Pow(x, n), Medium

  > ```pyhton
  >         if n < 0:
  >             x = 1 / x
  >             n = -n
  >         elif n == 0:
  >             return 1
  >         elif n == 1:
  >             return x
  >         elif n == 2:
  >             return x * x
  >         elif n == 3:
  >             return x * x * x
  >         ans = 1
  >         base = x
  >         while (n):
  >             if n%2 == 1:
  >                 ans = ans * base
  >             base = base *  base
  >             n = n // 2
  >         return ans
  > ```

- No.54 Spiral Matrix， Medium

  > ```python
  > # very slow, but save memory
  > # Runtime: 80 ms, faster than 6.46% Memory Usage: 29.9 MB, less than 100.00% 
  >     if matrix == [] or matrix == [[]]:
  >         return []
  >     elif len(matrix) == 1:
  >         return matrix[0]
  >     import numpy as np
  >     matrix = np.array(matrix)
  >     row, col = matrix.shape
  >     up, down, left, right = 0, row - 1, 0, col - 1
  >     res = []
  >     while up <= down and left <= right:
  >         res.extend(matrix[up, left : right + 1])
  >         res.extend(matrix[up + 1 : down + 1, right])
  >         if down > up:
  >             res.extend(matrix[down, left : right][::-1])
  >         if right > left:
  >             res.extend(matrix[up + 1 : down, left][::-1])
  >         up += 1
  >         down -= 1
  >         left += 1
  >         right -= 1
  >     return res
  > ```
  >
  > ```python
  > # rotate the matrix every time
  > 	def rotate(m):
  >         # zip(*m)将其转换为按列对应的迭代器
  >         # map()根据提供的函数对指定序列做映射，python3返回迭代器
  >         m = list(map(list, zip(*m)))  # ==> 转置操作！！！
  >         m.reverse()   # 上下颠倒，类似np.flipud()
  >         return m
  > 
  >     res = []
  >     while matrix:
  >         res += matrix[0]
  >         matrix = rotate(matrix[1:])
  >     return res
  > ```
- No 56 Merge Intervals, Medium
  > ```python
  >      intervals.sort()
  >      size = len(intervals)
  >      if size <= 1:
  >          return intervals
  >      ans = []
  >      start_pre, end_pre = intervals[0]
  >      for i in range(1, size):
  >          start, end = intervals[i]
  >          if start <= end_pre:
  >              end_pre = max(end, end_pre)
  >          else:
  >              ans.append([start_pre, end_pre])
  >              start_pre = start
  >              end_pre = end
  >      ans.append([start_pre, end_pre])
  >      return ans
  > ```
  

- No.59 Spiral Matrix II, Medium

- No.60 Permutation Sequence, Hard

  > ```python
  >     # faster than just use itertools.permutations()
  >     res = ''
  >     digits = [str(i + 1) for i in range(n)]
  >     t = k - 1
  >     for i in range(n, 0, -1):
  >         ind = t//math.factorial(i - 1)
  >         t%=math.factorial(i - 1)
  >         if t == 0:
  >             res += digits[ind] + "".join(digits[:ind] + digits[ind + 1:])
  >             return res
  >         else:
  >             res += digits[ind]
  >             del digits[ind]
  >     return res
  > ```



## **<span id="jump2">2 Linked list</span>**

  - No.2  Add Two Numbers, Medium

    > ***divide and conquer***

- No.19 Remove Nth Node From End of List, Medium

  > count the length and remove

- No.21 Merge Two Sorted Lists,  Easy

- No.23 Merge k Sorted Lists, Hard

  > connect all values , sort them,  and generate  ListNode

- No.24 Swap Nodes in Pairs, Medium

  > understand the movement of pointers

- No.25 Reverse Nodes in k-Group, Hard

  > ```python
  >         if k == 1:
  >             return head
  >         if head is None:
  >             return None
  > 
  >         p = ListNode(0)
  >         p.next = head
  >         pre = p
  >         curr = pre.next
  >         if curr is None:
  >             return head
  >         next_p = curr
  >         for i in range(k - 1):
  >             next_p = next_p.next
  >             if next_p is None:
  >                 return head
  > 
  >         while curr:
  > 
  >             pre.next = next_p
  >             curr_i = curr.next
  >             curr.next = next_p.next
  > 
  >             curr_pre = curr
  >             while curr_i != next_p:
  >                 temp = curr_i.next
  >                 curr_i.next = curr_pre
  >                 curr_pre = curr_i
  >                 curr_i = temp
  >             next_p.next = curr_pre
  > 
  >             pre = curr
  >             curr = curr.next
  >             next_p = curr
  >             if next_p is None:
  >                 return p.next
  >             for i in range(k - 1):
  >                 next_p = next_p.next
  >                 if next_p is None:
  >                     return p.next
  >         return p.next
  > ```

## **<span id="jump3">3 String</span>**

  - No.3  Longest Substring Without Repeating Characters, Medium
    > change start position while comparing the length of substring
    ```python
    	 def lengthOfLongestSubstring(self, s: str) -> int:
             position = {}
             max_length = 0
             start_pos = 0
             for i, c in enumerate(s):
                 if position.get(c) is not None and position[c] >= start_pos:
                     start_pos = position[c] + 1
                 position[c] = i
                 length = i - start_pos + 1
                 if max_length < length:
                     max_length = length
             return max_length
    ```

  - No.5 Longest Palindromic Substring, Medium

    > Odd and even; change max_length while comparing

    ```python
            if len(s) == 1 or s == s[::-1]:
                return s
            start_odd = 0
            start_even = 0
            max_odd = 1
            max_even = 0
            n = len(s)
            for i in range(n):
                odd = s[i - max_odd // 2 - 1: i + max_odd // 2 + 2]
                even = s[i - max_even: i + 2]
                while i - max_odd // 2 - 1 >= 0 and i + max_odd // 2 + 2 <= n and odd == odd[::-1]:
                    start_odd = i - max_odd // 2 - 1
                    max_odd += 2
                    odd = s[i - max_odd // 2 - 1: i + max_odd // 2 + 2]
                while i - max_even >= 0 and i + 2 <= n and even == even[::-1]:
                    start_even = i - max_even
                    max_even += 2
                    even = s[i - max_even: i + 2]
            if max_even < max_odd:
                return s[start_odd: start_odd + max_odd]
            else:
                return s[start_even: start_even + max_even]
    ```

- No.8 String to Integer (atoi), Medium

  > very easy to use regex

- No.10 Regular Expression Matching, Hard

  > just regex

- No.14 Longest Common Prefix, Easy



- No.30 Substring with Concatenation of All Words, Hard

  > ```python
  >         words_dict = {}
  >         word_num = len(words)
  >         for word in words:
  >             if word not in words_dict:
  >                 words_dict[word] = 1
  >             else:
  >                 words_dict[word] += 1
  >         word_len = len(words[0])
  >         res = []
  >         for i in range(len(s) + 1 - word_len * word_num):
  >             curr = {}
  >             j = 0
  >             while j < word_num:
  >                 word = s[i + j * word_len:i + j * word_len + word_len]
  >                 if word not in words:
  >                     break
  >                 if word not in curr:
  >                     curr[word] = 1
  >                 else:
  >                     curr[word] += 1
  >                 if curr[word] > words_dict[word]: break
  >                 j += 1
  >             if j == word_num:
  >                 res.append(i)
  >         return res
  > ```
  >
  
- No.43 Multiply Strings, Medium
- No.58 Length of Last Word, Easy





## 4 <span id="jump4">Sort</span> 

  - No.4 Median of Two Sorted Arrays, Hard

    > 1. converge into one array
    > 2. use two index (i, j) to save location which is being compared

- No.6 ZigZag Conversion Medium

  > ZigZag "PAYPALISHIRING", numRows=3 
  >
  > --> P      A     H     N  
  > 
  >       A  P  L  S  I   I   G  
  >    
  >       Y       I      R  
  > --> "PAHNAPLSIIGYIR"
  > -->row_id:  [0,1,2,1,0,1,2,1,0...]	
  > hint: import itertools

- No.7 Reverse Integer Easy

  > just reverse and verify validity 

- No.33 Search in Rotated Sorted Array, Medium

  > jump redundant numbers

- No.34 Find First and Last Position of Element in Sorted Array, Medium

  > O(logn) mid = (left + right) >> 1

- No.35 Search Insert Position, Easy

  > biscet



## 5 <span id="jump5">Stack</span> 

- No.20 Valid Parentheses, Easy

  > use stack

- No.32 Longest Valid Parentheses, Hard
  > 解题思路：返回括号串中合法括号串的长度。使用栈。这个解法比较巧妙，开辟一个栈，压栈的不是括号，而是未匹配左括号的索引！
  > ```python
  > max_len = 0
  > stack = []
  > last = -1
  > for i in range(len(s)):
  >     if s[i] == '(':
  >         stack.append(i)  # push the INDEX into the stack!!!!
  >     else:
  >         if stack == []:
  >             last = i
  >         else:
  >             stack.pop()
  >             if stack == []:
  >                 max_len = max(max_len, i - last)
  >             else:
  >                 max_len = max(max_len, i - stack[-1])
  > ```

## 6 <span id="jump6">DFS</span>

- No 21 Generate Parentheses, Medium
  > Method one: DFS
  > ```python
  >     def dfs(self, res, word, left, right):
  >         if left == 0 and right == 0:
  >             res.append(word)
  > 
  >         if left > 0 and right >= left:
  >             self.dfs(res, word + '(', left - 1, right)
  >         if right > 0 and right - 1 >= left:
  >             self.dfs(res, word + ')', left, right - 1)
  > 
  >     def generateParenthesis(self, n: int) -> List[str]:
  >         res = []
  >         self.dfs(res, "", n, n)
  >         return res
  > ```
  > Method two: from itertools import product
  > ```python
  >         from itertools import product
  >         ans = []
  >         for i in product(['(', ')'], repeat=(n - 1) * 2):  # generate Full Permutation
  >             temp = ''.join(i)
  >             count = 0
  >             for ch in temp:
  >                 if ch == '(':
  >                     count += 1
  >                 else:
  >                     count -= 1
  >                 if count < -1:
  >                     break
  >             if count == 0:
  >                 ans.append('(' + ''.join(i) + ')')
  >         return ans
  > ```

- No.37 Sudoku Solver

  > use dfs, validate now and future
  >
  > ```python
  >     def dfs(self, board: List[List[str]]):
  >         def is_valid(board: List[List[str]]):
  >             numbers = []
  >             for i, row in enumerate(board):
  >                 for j, c in enumerate(row):
  >                     if c != '.':
  >                         numbers += [(i, c), (c, j), (i // 3, j // 3, c)]
  >             return len(set(numbers)) == len(numbers)
  > 
  >         for col in range(9):
  >             for row in range(9):
  >                 if board[row][col] == '.':
  >                     for c in "123456789":
  >                         board[row][col] = c
  >                         if is_valid(board) and self.dfs(board):
  >                             return True
  >                         board[row][col] = '.'
  >                     return False
  >         return True
  > 
  >     def solveSudoku(self, board: List[List[str]]) -> None:
  >         self.dfs(board)
  > ```

- No.17 Letter Combinations of a Phone Number, Medium

  > use ***recursion***
  >
  > ```python
  > kv = {'2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl', '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz'}
  > size = len(digits)
  > if size == 0:
  > 	return []
  > elif size == 1:
  > 	return [c for c in kv[digits]]
  > temp = []
  > for i in range(size):
  > 	temp.append(kv[digits[i]])
  > 
  > words = []
  > next_words = self.letterCombinations(digits[1:])
  > for j in temp[0]:
  > 	for i in next_words:
  > 		words.append(j + i)
  > return words
  > ```

- No.39 Combination Sum, Medium

  > can repeatedly use Recursion
  >
  > ```python
  > def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
  >     res = []
  > 
  >     def dfs(set, target, idx):
  >         if target == 0:
  >             res.append(set)
  >             return
  >         for i in range(idx, len(candidates)):
  >             if candidates[i] <= target:
  >                 dfs(set + [candidates[i]], target - candidates[i], i)
  > 
  >     dfs([], target, 0)
  > 
  >     return res
  > ```

- No.40 Combination Sum, Medium

  > cannot repeatedly use
  >
  > dfs(set + [candidates[i]], target - candidates[i], i)  --> dfs(set + [candidates[i]], target - candidates[i], i+1)

- No.47 Permutations II, Medium

  > DFS
  >
  > ```python
  >         def backtrack(nums, path, out):
  >             if not nums:
  >                 out.append(path)
  >                 return
  >               
  >                        
  >             for x in range(len(nums)): 
  >                 # remove redundance
  >                 if x > 0:
  >                     if nums[x - 1] == nums[x]:
  >                         continue
  >                 
  >                 backtrack(nums[:x] + nums[x + 1:], path + [nums[x]], out) 
  >         
  >         out = [] 
  >         nums.sort()
  >         backtrack(nums , [], out)
  >         
  >         return out
  > ```
  
- No.51 N-Queens, Hard

  >  create four dictionary to save the flag whether it can put a Queen there.
  >
  > ```python
  >         if n == 1:
  >             return [["Q"]]
  >         elif n == 2:
  >             return []
  >         from collections import defaultdict
  >         board = [['.' for j in range(n)] for i in range(n)]
  >         rows = defaultdict(bool)
  >         cols = defaultdict(bool)
  >         diag1 = defaultdict(bool)  # rightup2leftdown
  >         diag2 = defaultdict(bool)  # leftup2rightdown
  > 
  >         def available(x, y):
  >             return not rows[x] and not cols[y] and not diag1[x+y] and not diag2[x-y]
  >         
  >         def update(x, y, flag):
  >             rows[x] = flag
  >             cols[y] = flag
  >             diag1[x+y] = flag
  >             diag2[x-y] = flag
  >             board[x][y] = 'Q' if flag==True else '.'
  >         
  >         def dfs(x):
  >             if x == n:
  >                 res.append([''.join(lst) for lst in board])
  >                 return
  >             for y in range(n):
  >                 if available(x , y):
  >                     update(x, y, True)
  >                     dfs(x+1)
  >                     update(x, y, False)       
  >                     
  >         res = []
  >         dfs(0)
  >         return res
  > ```
  >
  > 



## **<span id="jump7">7 DP</span>**

> Max problem

- No.32 Longest Valid Parentheses,  Hard

  > [hint](https://blog.csdn.net/qqxx6661/article/details/77876647)
  >
  > e.g "（( ) ( ))": [0 0 2 0 4 6]
  >
  > ```python
  > size = len(s)
  > if size < 2:
  >     return 0
  > dp = [0 for _ in range(size)]
  > for i in range(1, size):
  >     if s[i] == ')':
  >         j = i - 1 - dp[i - 1]   # 直接去查找前面的第j位移过了dp[i-1]位已经匹配的
  >         if j >= 0 and s[j] == '(':  # 如果那位是‘（’则可以总数多+2
  >             dp[i] = dp[i - 1] + 2
  >             if j - 1 >= 0:
  >                 dp[i] += dp[j - 1]  # 重点，会把这次匹配之前的加进去，例如（）（（））
  > return max(dp)
  > ```

- No.44 Wildcard Matching, Hard

  > Method one: analyze strings
  >
  > For each element in s
  > If *s==*p or *p == ? which means this is a match, then goes to next element s++ p++.
  > If p=='*', this is also a match, but one or many chars may be available, so let us save this *'s position and the matched s position.
  > If not match, then we check if there is a * previously showed up,
  >     if there is no *,  return false;
  >     if there is an *,  we set current p to the next element of *, and set current s to the next saved s position.
  >
  > ```python
  >         if p == s:
  >             return True
  >         elif p == "*":
  >             return True
  >         elif p == "" or s == "":
  >             return False
  > 
  >         i = 0
  >         size = len(p)
  >         # 合并重复的*
  >         while i < size - 1:
  >             if p[i] == "*" and p[i + 1] == "*":
  >                 p = p[:i + 1] + p[i + 2:]
  >                 size -= 1
  >             else:
  >                 i += 1
  >         i = 0
  >         j = 0
  >         p_star = -1
  >         s_pos = -1
  >         while i < len(s):
  > 
  >             if j < len(p) and (s[i] == p[j] or p[j] == '?'):
  >                 i += 1
  >                 j += 1
  >             elif j < len(p) and p[j] == "*":
  >                 p_star = j
  >                 j += 1
  >                 s_pos = i
  >             else:
  >                 if p_star >= 0:
  >                     j = p_star + 1
  >                     s_pos += 1
  >                     i = s_pos
  >                 else:
  >                     return False
  >         return (j == len(p)) or (j == len(p) - 1 and p[j] == "*")
  > ```
  >
  > Method two: DP
  >
  > ```python
  >         m = len(p)
  >         n = len(s)
  >         T = [[None] * (n + 1) for _ in range(m + 1)]
  > 
  >         # Recursion Base Cases
  >         # (1) p: "" matches s: ""
  >         T[0][0] = True 
  > 
  >         # (2) p: "" does not match non-empty s
  >         for j in range(1, n + 1):
  >             T[0][j] = False
  > 
  >         # (3) For T[i][0] = True, p must be '*', '**', '***', etc 
  >         #     Once p[i-1] != '*', all the T[i][0] afterwards will be False
  >         for i in range(1, m + 1):
  >             if p[i-1] == '*':
  >                 T[i][0] = T[i-1][0]
  >             else:
  >                 T[i][0] = False
  > 
  >         # Fill the table T (using recursion formulation -> see recursion code)      
  >         for i in range(1, m + 1):
  >             for j in range(1, n + 1):
  > 
  >                 pIdx = i - 1  # Because p length in m and last index is m - 1
  >                 sIdx = j - 1  # Because s length is n and last index is n - 1
  > 
  >                 # (i-1)th char matches (j-1)th char or (i-1)th char matches single char
  >                 if p[pIdx] == s[sIdx] or p[pIdx] == '?': 
  >                     T[i][j] = T[i-1][j-1]
  > 
  >                 # (i-1)th char matches any sequence of chars
  >                 elif p[pIdx] == '*':
  > 
  >                     # Recurrence relation in 2 variables: F(n,m) = F(n-1,m) + F(n,m-1)
  >                     T[i][j] = T[i-1][j] or T[i][j-1]
  > 
  >                 else:
  >                     T[i][j] = False
  > 
  >         return T[m][n]
  >     
  > ```
  >
  
- No 53 Maximum Subarray, Easy

  > ```python
  >     # dp save the max num which can be got until i-th number.
  >     # T = O(n), M = O(n)
  >     dp = [0]*len(nums)
  >     for i,num in enumerate(nums):            
  >         dp[i] = max(dp[i-1] + num, num)
  >     return max(dp)
  > 	
  >     # not dp, T=O(n), M=O(1)
  >     max_sum_until_i = max_sum= nums[0]
  >     for num in nums[1:]:
  >         max_sum_until_i = max(max_sum_until_i+num, num)
  >         max_sum = max(max_sum, max_sum_until_i)
  >         return max_sum
  > ```
  >
  > 
