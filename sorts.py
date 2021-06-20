from typing import List


class sort:
    def quick(self, nums:List[int]) -> List[int]:

        if len(nums) >= 2:
            base = nums[-1] # 选取基准值，可以为任何值
            left, right = [], []
            nums = nums[:-1]
            for num in nums:
                if num >= base:   # 大于等于基准值的数存于right
                    right.append(num)
                else:  # 小于基准值的数存于left
                    left.append(num)
            # print(left, '\t', base, '\t', right)
            return self.quick(left) + [base] + self.quick(right)
        else:
            return nums

    def quick_sort(self, nums: list, left: int, right: int) -> None:
        if left < right:
            i = left
            j = right
            # 取第一个元素为枢轴量
            pivot = nums[left]
            while i != j:
                # 交替扫描和交换
                # 从右往左找到第一个比枢轴量小的元素，交换位置
                while j > i and nums[j] > pivot:
                    j -= 1
                if j > i:
                    # 如果找到了，进行元素交换
                    # nums[i], nums[j] = nums[j], nums[i]
                    # break
                    nums[i] = nums[j]
                    i += 1
                # 从左往右找到第一个比枢轴量大的元素，交换位置
                while i < j and nums[i] < pivot:
                    i += 1
                if i < j:
                    nums[j] = nums[i]
                    j -= 1
            # 至此完成一趟快速排序，枢轴量的位置已经确定好了，就在i位置上（i和j)值相等
            nums[i] = pivot
            print(nums)
            # 以i为枢轴进行子序列元素交换
            self.quick_sort(nums, left, j-1)
            self.quick_sort(nums, j+1, right)		

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
    array = [2,3,5,1,1,4,6,15] 
    # array = [4,1,2,3,5] 
    print(array)  
    # print(s.merge_sort(array))
    s.quick_sort(array, 0, len(array)-1)
    print(array)


if __name__ == "__main__":
    main()

