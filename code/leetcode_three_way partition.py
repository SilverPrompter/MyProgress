def pivotArray(self, nums: List[int], pivot: int) -> List[int]:
    less = []
    equal = []
    greater = []
    for x in nums:
        if x > pivot:
            greater.append(x)
        elif x < pivot:
            less.append(x)
        else:
            equal.append(x)
    return less + equal + greater
