import numpy as np
import numpy.linalg as la


def pagerank(link_matrix, d):
    n = link_matrix.shape[0]
    J = np.ones((n, n))
    M = d * link_matrix + ((1 - d) / n) * J

    r = 100 * np.ones(n) / n
    r_prev = r
    r_next = M @ r

    while la.norm(r_prev - r_next) > 0.01:
        r_prev = r_next
        r_next = M @ r_next

    return r_next


num = int(input())
d = 0.5
websites = input().split()

lst = []

for _ in range(num):
    lst.append([float(x) for x in input().split()])

L = np.array(lst).reshape(num, num)
query = input()

r = pagerank(L, d)
args = np.argsort(r)[::-1][:5]

if query in set(websites):
    print(query)
    print(*[websites[i] for i in args if websites[i] != query], sep='\n')
else:
    print(*[websites[i] for i in args], sep='\n')
