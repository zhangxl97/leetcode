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
    
    def merge(self, a, b):
        c = []
        h = j = 0
        while j < len(a) and h < len(b):
            if a[j] < b[h]:
                c.append(a[j])
                j += 1
            else:
                c.append(b[h])
                h += 1

        if j == len(a):
            for i in b[h:]:
                c.append(i)
        else:
            for i in a[j:]:
                c.append(i)

        return c


    def merge_sort(self, lists):
        if len(lists) <= 1:
            return lists
        middle = len(lists)//2
        left = self.merge_sort(lists[:middle])
        right = self.merge_sort(lists[middle:])
        return self.merge(left, right)
    

def main():
    s = sort()
    array = [2,3,5,7,1,4,6,15,5,2,7,9,10,15,9,17,12]  
    print(array)  
    print(s.merge_sort(array))


if __name__ == "__main__":
    main()

