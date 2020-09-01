# Notes

## Index

> Github: https://github.com/zhangxl97/leetcode.git

1. [Number Problems](#jump1)
2. [Linked list](#jump2)
3. [String](#jump3)
4. [Sort](#jump4)
5. [Stack](#jump5)

## **<span id="jump1">1. Number Problems</span>**

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

- No.12. Integer to Roman, Medium and No.13 Roman to Integer,  Easy

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

- No.18. 4Sum

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
  > 

## **<span id="jump2">2. Linked list</span>**

  - No. 2  Add Two Numbers, Medium

    > ***divide and conquer***

- No.19. Remove Nth Node From End of List, Medium

  > count the length and remove

## **<span id="jump3">3. String</span>**

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

## 4. <span id="jump4">Sort</span> 

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

## 5. <span id="jump5">Stack</span> 

- No.20 Valid Parentheses, Easy

  > use stack