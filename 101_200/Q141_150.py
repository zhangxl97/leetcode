from typing import List


class Solution:
    # 141, 逆波兰表达式求值, Medium
    def evalRPN(self, tokens: List[str]) -> int:
        
        stack = []
        for token in tokens:
            if token == "+":
                oprand2 = stack.pop()
                oprand1 = stack.pop()

                stack.append(oprand1 + oprand2)

            elif token == "*":
                oprand2 = stack.pop()
                oprand1 = stack.pop()

                stack.append(oprand1 * oprand2)
                
            elif token == "/":
                oprand2 = stack.pop()
                oprand1 = stack.pop()

                stack.append(int(oprand1 / oprand2))
            
            elif token == "-":
                
                oprand2 = stack.pop()
                oprand1 = stack.pop()

                stack.append(oprand1 - oprand2)
            else:
                stack.append(int(token))
            # print(stack)
        
        return stack[0]
                
            

def main():
    s = Solution()

    print(s.evalRPN(tokens = ["10","6","9","3","+","-11","*","/","*","17","+","5","+"]))


if __name__ == "__main__":
    main()

