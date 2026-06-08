    def areSimilar(self, mat: List[List[int]], k: int) -> bool:
        mat_o = [row[:] for row in mat]
        n = len(mat[0])
        if k % n == 0:
            return True
        for x, y in enumerate(mat):
            if x % 2 == 0:
                mat[x] = y[k%n:] + y[:k%n]
            else:
                mat[x] = y[-k%n:] + y[:-k%n]
        return mat == mat_o
