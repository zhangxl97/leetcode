from typing import List


class sort:
    def quick(self, nums:List[int]) -> List[int]:

        if len(nums) >= 2:
            base = nums[-1] # 选取基准值，可以为任何值
            left, right = [], []
            nums.remove(base)
            for num in nums:
                if num >= base:   # 大于等于基准值的数存于right
                    right.append(num)
                else:  # 小于基准值的数存于left
                    left.append(num)
            # print(left, '\t', base, '\t', right)
            return self.quick(left) + [base] + self.quick(right)
        else:
            return nums
    

def main():
    s = sort()
    array = [2,3,5,7,1,4,6,15,5,2,7,9,10,15,9,17,12]  
    print(array)  
    print(s.quick(array))


if __name__ == "__main__":
    main()

