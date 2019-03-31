import numpy as np
import matplotlib.pyplot as plt

file = 'theta_q'
theta = np.load('thetas/{}.npy'.format(file))
print('Theta hat shape', np.shape(theta), 'und lautet:')
print(theta)
plt.figure(figsize = (18, 5))
plt.imshow(theta, cmap = 'gray_r')
plt.colorbar(shrink = 0.6)
#plt.savefig('{}.png'.format(file), format= 'png')
plt.show()

#q_data = np.load('q_data/q_data.npy')
#print('q_data hat shape', np.shape(q_data), 'und lautet:')
#print(q_data[-1])

#theta2 = np.load('thetas/{}.npy'.format('theta_q_best_container_nach_pizza_waehrend_fussball'))
theta_mit_bomben = np.load('thetas/{}.npy'.format('theta_q_mit_bomben')
#plt.figure(figsize = (18, 5))
#plt.imshow(theta2-theta, cmap = 'gray_r')
#plt.colorbar(shrink = 0.6)
#plt.savefig('{}_diff.png'.format(file), format= 'png')

theta_mit_bomben[5, 32] = 3
#theta2[4, 31] = -10
np.save('thetas/theta_q_mit_bomben.npy', theta_mit_bomben)
