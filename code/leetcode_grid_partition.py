    def canPartitionGrid(self, grid: List[List[int]]) -> bool:
        total = sum(sum(row) for row in grid)
        if total % 2 != 0:
            return False
    
        running = 0
        for x, y in enumerate(grid):
            running += sum(y)
            if running == total / 2 and x != len(grid) - 1:
                return True
        
        running = 0
        for x in range(len(grid[0])):
            running += sum(grid[i][x] for i in range(len(grid)))
            if running == total / 2 and x != len(grid[0]) - 1:
                return True
        
        return False
