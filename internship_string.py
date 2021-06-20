from typing import List

class Solution:
    # 242. 有效的字母异位词, Easy
    def isAnagram(self, s: str, t: str) -> bool:
        # chars = {}
        # for c in s:
        #     if chars.get(c) is None:
        #         chars[c] = [1, 0]
        #     else:
        #         chars[c][0] += 1
        # for c in t:
        #     if chars.get(c) is None:
        #         return False
        #     else:
        #         chars[c][1] += 1
        # for key in chars:
        #     if chars[key][0] != chars[key][1]:
        #         return False
        # return True
        cnts = [0] * 26
        for c in s:
            cnts[ord(c) - 97] += 1
        for c in t:
            cnts[ord(c) - 97] -= 1
        for cnt in cnts:
            if cnt != 0:
                return False
        return True

    # 409. 最长回文串, Easy
    def longestPalindrome(self, s: str) -> int:
        from collections import Counter
        s = Counter(s)
        odds = 0
        ans = 0
        for c in s:
            if s[c] % 2 == 1:
                odds += 1
            ans += s[c]
        return ans - ((odds - 1) if odds > 1 else 0)

    # 205. 同构字符串, Easy
    def isIsomorphic(self, s: str, t: str) -> bool:
        # kvs = {}
        # visited = {}
        # for i in range(len(s)):
        #     if kvs.get(s[i]) is None:
        #         if visited.get(t[i]) is not None:
        #             return False
        #         kvs[s[i]] = t[i]
        #         visited[t[i]] = True
        #     else:
        #         if kvs[s[i]] != t[i]:
        #             return False
        # return True
        pre_exist_s = [0] * 256
        pre_exist_t = [0] * 256

        for i in range(len(s)):
            cs, ct = s[i], t[i]
            if pre_exist_s[ord(cs)] != pre_exist_t[ord(ct)]:
                return False
            pre_exist_s[ord(cs)] = i + 1
            pre_exist_t[ord(ct)] = i + 1
        return True

    # 647. 回文子串, Medium
    def countSubstrings(self, s: str) -> int:

        # size = len(s)
        # cnt = 0
        # for i in range(size):
        #     for j in range(i, size):
        #         if s[i:j+1]==s[i:j+1][::-1]:
        #             cnt += 1
        # return cnt

        def helper(s, start, end):
            while start >= 0 and end < len(s) and s[start] == s[end]:
                start -= 1
                end += 1
                self.cnt += 1
            
        
        self.cnt = 0
        for i in range(len(s)):
            helper(s, i, i)  # 奇数长度
            helper(s, i, i + 1)  # 偶数长度
        
        return self.cnt

    # 9. 回文数, Easy
    def isPalindrome(self, x: int) -> bool:

        if x == 0:
            return True
        if x < 0 or x % 10 == 0:
            return False
        # x = str(x)
        # return x == x[::-1]

        right = 0
        while x > right:
            right = right * 10 + x % 10
            x = x // 10
        return x == right or x == right // 10

    # 696. 计数二进制子串, Easy ⭐
    def countBinarySubstrings(self, s: str) -> int:

        # O(N^2)  TLE
        # ans = 0
        # for i in range(len(s) - 1):
        #     flag = False
        #     cnt = [0, 0]
        #     cnt[int(s[i])] = 1
        #     for j in range(i + 1, len(s)):
        #         if s[j] != s[j - 1]:
        #             if flag is False:
        #                 flag = True
        #             else:
        #                 break
        #         cnt[int(s[j])] += 1
        #         if cnt[0] == cnt[1]:
        #             ans += 1
        # return ans

        # O(N)
        pre_len = 0
        cur_len = 1
        cnt = 0
        for i in range(1, len(s)):
            if s[i] == s[i - 1]:
                cur_len += 1
            else:
                pre_len = cur_len
                cur_len = 1
            
            if pre_len >= cur_len:
                cnt += 1
        return cnt


                



def main():
    s = Solution()

    # print(s.isAnagram(s = "rat", t = "car"))
    # print(s.longestPalindrome("badb"))
    # print(s.isIsomorphic(s = "ab", t = "aa"))
    # print(s.countSubstrings("aaa"))
    # print(s.isPalindrome(121))
    print(s.countBinarySubstrings("10101"))

if __name__ == "__main__":
    main()
