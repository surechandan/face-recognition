def main():
    n = int(input("Enter number of line segments: "))
    segs = []
    coord_to_idx = {}
    idx = 0
    edges = 0

    for _ in range(n):
        x1, y1, x2, y2 = map(int, input("Enter x1 y1 x2 y2: ").split())
        if x1 == x2 and y1 == y2:
            continue
        a = (x1, y1)
        b = (x2, y2)
        if a not in coord_to_idx:
            coord_to_idx[a] = idx; idx += 1
        if b not in coord_to_idx:
            coord_to_idx[b] = idx; idx += 1
        u = coord_to_idx[a]
        v = coord_to_idx[b]
        segs.append((u, v))
        edges += 1
