def pathExistenceQueries(self, n: int, nums: List[int], maxDiff: int, queries: List[List[int]]) -> List[bool]:
    group = [0]
    for x in range(1, n):
        if nums[x] - nums[x-1] <= maxDiff:
            group.append(group[x-1])
        else:
            group.append(group[x-1] + 1)

    answer = []
    for u, v in queries:
        answer.append(group[u] == group[v])
    return answer
