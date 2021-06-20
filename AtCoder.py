# -*- coding: utf-8 -*-
def sample():
    # 整数の入力
    a = int(input())
    # スペース区切りの整数の入力
    b, c = map(int, input().split())
    # 文字列の入力
    s = input()
    # 出力
    print("{} {}".format(a+b+c, s))

# QA
# def S(x):
#     tmp = 0
#     while x > 0:
#         tmp += x%10
#         x = x // 10
#     return tmp

# num1, num2 = map(int, input().split())
# S1, S2 = S(num1), S(num2)
# if S1 >= S2:
#     print(S1)
# else:
#     print(S2)

# QB
# def slope(p1, p2):
#     return (p2[1] - p1[1]) / (p2[0] - p1[0])


# N = int(input())
# if N <= 1:
#     x, y = map(int, input().split())
#     print(0)
# else:
#     points = []
#     for i in range(N):
#         x, y = map(int, input().split())
#         points.append([x, y])
#     cnt = 0
#     for i in range(0, N - 1):
#         for j in range(i + 1, N):
#             if -1 <= slope(points[i], points[j]) <= 1:
#                 cnt += 1
#     print(cnt) 

# QC
# N = int(input())
# words = []
# bad_word = None
# for i in range(N):
#     words.append(input())
# words = sorted(list(set(words)))
# word_dict = {}
# for word in words:
#     if word_dict.get(word) is None:
#         if word[0] == "!":
#             if word_dict.get(word[1:]) is not None:
#                 bad_word = word[1:]
#                 break
#             else:
#                 word_dict[word] = True
#         else:
#             if word_dict.get("!" + word) is not None:
#                 bad_word = word
#                 break
#             else:
#                 word_dict[word] = True
        
# if bad_word is not None:
#     print(bad_word)
# else:
#     print("satisfiable")

# QD
# SUM(lamda_i * (Ai + Bi)) > SUM((1 - lamda_i) * Ai)
N = int(input())
cities = []
sums = []
aoki = 0
for i in range(N):
    Ai, Bi = map(int, input().split())
    aoki += Ai
    cities.append([Ai, Bi])
    sums.append(2 * Ai + Bi)
sums.sort(reverse=True)
cnt = 0
counts = 0
for i in range(N):
    cnt += 1
    counts += sums[i]
    if counts > aoki:
        print(cnt)
        break
