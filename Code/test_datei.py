import numpy as np

the = np.load('./agent_code/cbt_agent/thetas/theta_q.npy')
print(the)
import matplotlib.pyplot as plt
plt.imshow(the, cmap = 'gray_r')
plt.colorbar()
plt.show()
