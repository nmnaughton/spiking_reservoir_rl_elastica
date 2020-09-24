from sklearn.linear_model import LinearRegression, Ridge
import numpy as np
import matplotlib.pyplot as plt


def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


npzfile = np.load('state_action_pairs.npz')
time = npzfile['time_val']
iter_num = npzfile['iter_num']
state  = npzfile['state']
action = npzfile['action']
reward = npzfile['reward']
score  = npzfile['score']


RC_state = np.load('reservoir_outputs.npy')


window = 10
smooth_action = np.zeros((moving_average(action[:,0],window).shape[0],action.shape[1]))

for i in range(action.shape[1]):
    smooth_action[:,i] = moving_average(action[:,i],window)


smooth_RC_state = np.zeros((moving_average(RC_state[:,0],window).shape[0],RC_state.shape[1]))
for i in range(RC_state.shape[1]):
    smooth_RC_state[:,i] = moving_average(RC_state[:,i],window)



start = 0
stop = int(len(smooth_RC_state)*0.5)
print('stop:',stop)
reg_smooth = LinearRegression().fit(smooth_RC_state[start:stop], smooth_action[start:stop])
print("R^2 on training data:", reg_smooth.score(smooth_RC_state[start:stop], smooth_action[start:stop]))
print("R^2 on all of data:", reg_smooth.score(smooth_RC_state, smooth_action))
print("R^2 on testing data:", reg_smooth.score(smooth_RC_state[stop:], smooth_action[stop:]))

smooth_test_data = reg.predict(smooth_RC_state[:30000])

tr_size = len(X_train)
plt.figure(figsize=(32, 4))
plt.xlim(0, len(smooth_test_data))
plt.plot(smooth_action[:30000,:], 'b', linewidth = 0.5, label='data')
plt.plot(smooth_test_data, 'r', linewidth = 0.5, alpha=0.9, label='predict')
plt.axvspan(0, stop, facecolor='g', alpha=0.1, label =  "train area")
plt.legend(loc='upper right')
plt.ylim([-2,2])
