from typing import List


# 232. 用栈实现队列, Easy
class MyQueue:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.elements = []
        # self.size = 0


    def push(self, x: int) -> None:
        """
        Push element x to the back of queue.
        """
        self.elements.append(x)
        # self.size += 1


    def pop(self) -> int:
        """
        Removes the element from in front of queue and returns that element.
        """
        # self.size -= 1
        return self.elements.pop(0)


    def peek(self) -> int:
        """
        Get the front element.
        """
        return self.elements[0]


    def empty(self) -> bool:
        """
        Returns whether the queue is empty.
        """
        return self.elements == []

# 155. 最小栈, Easy
class MinStack:

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.elements = []
        self.min_val = float('inf')
        self.min_stack = []


    def push(self, val: int) -> None:
        self.elements.append(val)
        self.min_val = min(self.min_val, val)
        self.min_stack.append(self.min_val)

    def pop(self) -> None:
        self.min_stack.pop()
        self.min_val = self.min_stack[-1] if self.min_stack else float('inf')
        return self.elements.pop()


    def top(self) -> int:
        return self.elements[-1]


    def getMin(self) -> int:
        return self.min_val

class Solution:
    # 20. 有效的括号, Easy
    def isValid(self, s: str) -> bool:
        stack = []
        for c in s:
            if c == "(" or c == "[" or c == "{":
                stack.append(c)
            elif stack == []:
                return False
            else:
                past = stack.pop()
                if c == ")":
                    if past != "(":
                        return False
                elif c == "]":
                    if past != "[":
                        return False
                elif c == "}":
                    if past != "{":
                        return False
        return stack == []

    # 739. 每日温度, 
    # 单调栈
    def dailyTemperatures(self, T: List[int]) -> List[int]:

        ans = [0] * len(T)

        indexes = []

        for index, t in enumerate(T):

            while indexes != [] and t > T[indexes[-1]]:
                pre_index = indexes.pop()
                ans[pre_index] = index - pre_index
            indexes.append(index)
        return ans

    # 503. 下一个更大元素 II, Medium
    def nextGreaterElements(self, nums: List[int]) -> List[int]:

        ans = [-1] * len(nums)

        nums = nums + nums

        indexes = []

        for index, num in enumerate(nums):
            while indexes != [] and num > nums[indexes[-1]]:
                pre_index = indexes.pop()
                if pre_index >= (len(nums) // 2):
                    pre_index = pre_index % (len(nums) // 2)
                ans[pre_index] = num
            indexes.append(index)
        return ans



def main():

    # obj = MyQueue()
    # obj.push(1)
    # obj.push(2)
    # param_2 = obj.pop()
    # param_3 = obj.peek()
    # param_4 = obj.empty()
    # print(param_2)
    # print(param_3)
    # print(param_4)

    # minStack = MinStack()
    # minStack.push(-2)
    # minStack.push(0)
    # minStack.push(-3)
    # print(minStack.getMin())
    # minStack.pop()
    # print(minStack.top())
    # print(minStack.getMin())
    # minStack.pop()
    # minStack.pop()

    s = Solution()

    # print(s.isValid(s = "{[]}"))
    # print(s.dailyTemperatures( [73, 74, 75, 71, 69, 72, 76, 73]))
    # print(s.nextGreaterElements([1,2,1]))


if __name__ == "__main__":
    main()
