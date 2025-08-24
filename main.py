sizes = [2, 3, 1]

print(
    [(x, y) for x, y in zip(sizes[:-1], sizes[1:])]
)