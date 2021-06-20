from typing import List
from tree import TreeNode


class Solution:
    # 241, Medium
    def diffWaysToCompute(self, expression: str) -> List[int]:
        
        ans = []
        for i, c in enumerate(expression):
            if c == "+" or c == "-" or c == "*":
                left = self.diffWaysToCompute(expression[: i])
                right = self.diffWaysToCompute(expression[i + 1:])

                for l in left:
                    for r in right:
                        if c == "+":
                            ans.append(l + r)
                        elif c == "-":
                            ans.append(l - r)
                        else:
                            ans.append(l * r)
        if ans == []:
            ans = [int(expression)]
        return ans

    # 95
    def generateTrees(self, n: int) -> List[TreeNode]:

        def generate(nodes):
            if len(nodes) == 0:
                return [None]
            
            ans = []

            for i, node in enumerate(nodes):
                lefts = generate(nodes[:i])
                rights = generate(nodes[i+1:])
                for left in lefts:
                    for right in rights:
                        root = TreeNode(node)
                        root.left = left
                        root.right = right
                        ans.append(root)

            return ans

        
        nodes = [i + 1 for i in range(n)]
        return generate(nodes)




def main():
    s = Solution()

    # print(s.diffWaysToCompute("2*3-4*5"))
    roots = s.generateTrees(3)
    for root in roots:
        # root.BFS(root)
        root.BFS(root)
        print(root.maxDepth(root))
        print("")


if __name__ == "__main__":
    main()
