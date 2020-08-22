# Notes

## Index

1. [Number Problems](#jump1)
2. [Linked list](#jump2)
3. [String](#jump3)
4. [Sort](#jump4)

## **<span id="jump1">1. Number Problems</span>**

- No.1  Two Sum, Easy

  > confirm whether target - n is in the rest list
  >
  > use dictionary

- No.9  Palindrome Number, Easy

  > str(x) == str(x)[::-1]

## **<span id="jump2">2. Linked list</span>**

  - No. 2  Add Two Numbers, Medium

    > divide and conquer

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

## 4. <span id="jump4">Sort</span> 

  - No.4 Median of Two Sorted Arrays, Hard

    > 1. converge into one array
    > 2. use two index (i, j) to save location which is being compared

- No.6 ZigZag Conversion Medium

  > ZigZag "PAYPALISHIRING", numRows=3 
  >
  > --> P      A     H     N    --> "PAHNAPLSIIGYIR"
  >       A  P  L  S  I   I   G
  >       Y       I      R
  >
  > -->row_id:  [0,1,2,1,0,1,2,1,0...]	
  >
  > hint: import itertools

- No.7 Reverse Integer Easy

  > just reverse and verify validity 