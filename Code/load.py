import numpy as np

cont = []
with open('npload.txt', 'r') as f:
    for line in f:
        cont.append(line)
print(len(cont))
print(cont)

a = np.loadtxt('npload.txt')
print(np.shape(a))
print(len(a))

a[-1] = np.array([1, 2, 3, 4])
np.savetxt('test_output_np.txt', a, fmt = '%i')
with open('test_output.txt', 'w') as g:
    g.write(str(a))