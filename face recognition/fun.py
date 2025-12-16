def reflect(direction, mirror):
    # direction: 0=up,1=right,2=down,3=left
    if mirror == '/':
        return [1, 0, 3, 2][direction]  # turn accordingly
    elif mirror == '\\':
        return [3, 2, 1, 0][direction]
    return direction

def max_loop(grid, M, N):
    directions = [(-1,0),(0,1),(1,0),(0,-1)]  # up, right, down, left
    max_len = 0

    for i in range(M):
        for j in range(N):
            if grid[i][j] in ['/', '\\']:  # start only at mirrors
                for d in range(4):  # try all directions
                    visited = {}
                    r, c, dirn = i, j, d
                    step = 0
                    while 0 <= r < M and 0 <= c < N:
                        if (r,c,dirn) in visited:
                            max_len = max(max_len, step - visited[(r,c,dirn)])
                            break
                        visited[(r,c,dirn)] = step
                        step += 1
                        if grid[r][c] in ['/', '\\']:
                            dirn = reflect(dirn, grid[r][c])
                        dr, dc = directions[dirn]
                        r, c = r+dr, c+dc
    return max_len

# Driver
M, N = map(int, input().split())
grid = [input().split() for _ in range(M)]
print(max_loop(grid, M, N))
