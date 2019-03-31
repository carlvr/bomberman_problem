import numpy as np
from time import sleep
from scipy.optimize import curve_fit
import sys 
import traceback
import random
import copy

moves = np.array([[]])
path = '' #'./agent_code/cbt_agent/'
crate_counter = 0
round_number = 1
scores = np.zeros((1, 4))

def func_curve(X, a_1, a_2, a_3, a_4, a_5, a_6, a_7, a_8, a_9, a_10, a_11, a_12, a_13, a_14, a_15, a_16, a_17, a_18, a_19, a_20, a_21, a_22, a_23, a_24, a_25, a_26, a_27, a_28, a_29, a_30, a_31, a_32, a_33, a_34):
    (x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19, x_20, x_21, x_22, x_23, x_24, x_25, x_26, x_27, x_28, x_29, x_30, x_31, x_32, x_33) = X
    return a_1*x_1 + a_2*x_2 + a_3*x_3 + a_4*x_4 + a_5*x_5 + a_6*x_6 + a_7*x_7 + a_8*x_8 + a_9*x_9 + a_10*x_10 + a_11*x_11 + a_12*x_12 + a_13*x_13 + a_14*x_14 + a_15*x_15 + a_16*x_16 + a_17*x_17 + a_18*x_18 + a_19*x_19 + a_20*x_20 + a_21*x_21 + a_22*x_22 + a_23*x_23 + a_24*x_24 + a_25*x_25 + a_26*x_26 + a_27*x_27 + a_28*x_28 + a_29*x_29 + a_30*x_30 + a_31*x_31 + a_32*x_32 + a_33*x_33 + a_34


def positions(self):
    agent = self.game_state['self']
    agent_pos = np.array([agent[0], agent[1]])
    coins = np.array(self.game_state['coins'])
    if len(coins) == 0:
        # return agent_pos three times, otherwise the agent would try to go to the top left corner, this way he "tries to stay where he is"
        return agent_pos, agent_pos, np.zeros(2)
    else:
        dist = np.array([])   
        for i in range(len(coins)):
            dist = np.append(dist, np.linalg.norm(agent_pos - coins[i]))
        coin = coins[np.argmin(dist)]
        mean_coin = []
        
        for i in range(len(coins)):
            if dist[i] != 0:
                mean_coin.append((agent_pos - coins[i])/dist[i]**(3/2))
        if (len(np.shape(mean_coin)) > 1):
            mean_coin = np.sum(mean_coin, axis=0)
        if np.linalg.norm(mean_coin) != 0: 
            mean_coin = (mean_coin)/ np.linalg.norm(mean_coin) 
        if (mean_coin == []):
            mean_coin = np.zeros(2)
        return agent_pos, np.array(coin), mean_coin # agent_pos, absolute pos naechster coin, richtung mean coin

def difference(self):
    agent_pos, coin, mean_coin = positions(self) 
    arena = self.game_state['arena']
    diff = np.linalg.norm(agent_pos - coin)
    if diff != 0 :
        direction = (agent_pos - coin) / diff
    else:
        direction = np.zeros(2)

    return np.array([diff, *direction, *mean_coin])


def crate_positions(self):
    agent = self.game_state['self']
    agent_pos = np.array([agent[0], agent[1]])
    arena = self.game_state['arena']
    crates = []
    for i in range(np.shape(arena)[0]):
        for j in range(np.shape(arena)[1]):
            if arena[i][j] == 1:
                crates.append([i,j])
    crates = np.array(crates) 
    if len(crates) == 0:
        # return agent_pos three times, otherwise the agent would try to go to the top left corner, this way he "tries to stay where he is"
        return agent_pos, agent_pos, np.zeros(2)
    else:
        dist = np.array([])   
        for i in range(len(crates)):
            dist = np.append(dist, np.linalg.norm(agent_pos - crates[i]))
        crate = crates[np.argmin(dist)]
        mean_crate = []
        
        for i in range(len(crates)):
            if dist[i] != 0:
                mean_crate.append((agent_pos - crates[i])/dist[i]**(3/2))
        if (len(np.shape(mean_crate)) > 1):
            mean_crate = np.sum(mean_crate, axis=0)
        if np.linalg.norm(mean_crate) != 0: 
            mean_crate = (mean_crate)/ np.linalg.norm(mean_crate) 
        if (mean_crate == []):
            mean_crate = np.zeros(2)
        return agent_pos, np.array(crate), mean_crate # agent_pos, absolute pos naechster coin, richtung mean coin

def crate_diff(self):
    agent_pos, crate, mean_crate = crate_positions(self)     
    diff = np.linalg.norm(agent_pos - crate)
    if diff != 0 :
        direction = (agent_pos - crate) / diff
        if diff != 1:
            diff = 0
    else:
        direction = np.zeros(2)

    return np.array([*direction, *mean_crate, diff])

def check_even_odd(position):
    # 1 means, that the row/column is free
    # x and y are swapped, because the y direction is free if the x value is odd
    position = np.array(position)[np.r_[1, 0]] #np.r_ does the swap of x and y
    return position % 2
   
def last_move(self):
    forbidden = np.zeros(4)
    try:
        last_move_ = self.events[0]
    except:
        last_move_ = -1
    if (last_move_ in [0, 1, 2, 3]):
        forbidden[int(last_move_ + 1 - 2 * (last_move_ % 2))] = 1
    return forbidden


def own_bomb_ticking(self):
    #returns 1 if own bomb is ticking, else 0
    bomb_possible = self.game_state['self'][3]
    return (bomb_possible + 1) % 2

def explosion_radius_single_bomb(coordinates):
    check_free = check_even_odd(coordinates)
    x, y, x_, y_ = *coordinates, *check_free
    row = [[item, y] for item in np.unique(np.clip(np.r_[x-3:x+4], 0, 16))] if x_ else [[x, y]]
    column = [[x, item] for item in np.unique(np.clip(np.r_[y-3:y+4], 0, 16))] if y_ else [[x, y]] 
    return row + column

def position_in_danger(position, self):
    danger = 0
    for bomb in self.game_state['bombs']:
        if (list(position) in explosion_radius_single_bomb(bomb[:2])):
            if bomb[2] == 0:
                danger = 5
            else:
                danger = np.max([(0.633-0.133*bomb[2]), danger])
    if self.game_state['explosions'][tuple(position)] == 2:
        danger = 5
    return danger + 4 * np.clip(danger, 0, 0.1)

def number_of_crates_in_explosion_radius(self):
    own_pos = np.array(self.game_state['self'][:2])
    explosion_radius = explosion_radius_single_bomb(own_pos)
    tiles = np.array([self.game_state['arena'][tuple(item)] for item in explosion_radius])
    number = len(np.arange(len(tiles))[tiles == 1])
    number = 0.7 * number if number <= 2 else number
    return number

def next_move_danger(self):
    # returns 0 if no danger is in the corresponding direction or a wall 
    danger = np.zeros(4)
    own_pos = np.array(self.game_state['self'][:2])
    pos_diffs = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
    for i in range(4):
        danger[i] = position_in_danger(own_pos + pos_diffs[i], self)
        pos = own_pos + pos_diffs[i]
        if (np.abs(self.game_state['arena'][tuple([pos[0], pos[1]])]) == 1):
            danger[i] = 0.8
    return danger

def directions_blocked(pos, self):
    pos = np.array(pos)
    # returns 0 if direction is free, 1 if it is blocked 
    blocked = np.zeros(4)
    pos_diffs = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
    for i in range(4):
        blocked[i] = np.abs(self.game_state['arena'][tuple(pos + pos_diffs[i])])
        if list(pos + pos_diffs[i]) in [list(coord[:-1]) for coord in self.game_state['bombs']]:
            blocked[i] = 1
        if list(pos + pos_diffs[i]) in [list(coord[:2]) for coord in self.game_state['others']]:
            blocked[i] = 1
    return blocked
    
def next_move_blocked (self):
    blocked = np.zeros(4)
    own_pos = np.array(self.game_state['self'][:2])
    pos_diffs = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]]) 
    for i in range(4):
        current_pos = np.array(own_pos + pos_diffs[i])
        if (current_pos < 1).any() or (current_pos > 15).any():
            blocked[i] = 1
        else:
            blocked[i] = -3 + np.clip(np.sum(directions_blocked(current_pos, self)), 3, 4)
    return blocked

def tile_blocked(pos, self):
    blocked = np.abs(self.game_state['arena'][tuple(pos)])
    if list(pos) in [list(coord[:-1]) for coord in self.game_state['bombs']]:
        blocked = 1
    if list(pos) in [list(coord[:2]) for coord in self.game_state['others']]:
        blocked = 1
    return blocked

def no_through_road(self, check_no_bomb = False):
    blocked = np.zeros(4)
    own_pos = np.array(self.game_state['self'][:2])
    if (list(own_pos) not in [list(coord[:-1]) for coord in self.game_state['bombs']] and not check_no_bomb):
        return blocked
    pos_diffs = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]]) # left, right, up, down 
    orthogonal_list = np.array([[2, 3], [0, 1]])
    for i in range(4): #all directions
        cur_orthogonal = orthogonal_list[int(np.floor(0.5*i))]
        for j in range(4):
            current_pos = own_pos + (j+1)*pos_diffs[i]
            if (tile_blocked(current_pos, self) == 1):
                blocked[i] = 1
                break
            elif (directions_blocked(current_pos, self)[cur_orthogonal] == 0).any():
                break
    return blocked

def no_bomb(self):
    all_directions_blocked = no_through_road(self, check_no_bomb = True).all()
    return 1 if all_directions_blocked else 0

def killable_opponents(self):
    number = 0
    if own_bomb_ticking(self) == 0:
        explosion_radius = explosion_radius_single_bomb(self.game_state['self'][:2])
        for opponent in self.game_state['others']:
            if list(opponent[:2]) in explosion_radius:
                number += 1
    return number


def q_function(theta_q, features):
    f = theta_q[:,-1]
    for i in range(len(features)):
        f = f + theta_q[:, i] * features[i] 
    return f


def build_features (self):
    features = difference(self) # [diff, *direction, *mean_coin] also 5 Werte, Indizes 0, 1, 2, 3, 4
    features = np.append(features, check_even_odd(self.game_state['self'][:2])) # 2 Werte, Indizes 5, 6
    features = np.append(features, last_move(self)) # 4 Werte, Indizes 7, 8, 9, 10
    features = np.append(features, position_in_danger(self.game_state['self'][:2], self)) # 1 Wert, Index 11
    features = np.append(features, own_bomb_ticking(self)) # 1 Wert, Index 12
    features = np.append(features, number_of_crates_in_explosion_radius(self)) # 1 Wert, Index 13
    features = np.append(features, directions_blocked(self.game_state['self'][:2], self)) # 4 Werte, Indizes 14, 15, 16, 17
    features = np.append(features, next_move_danger(self)) # 4 Werte, Indizes 18, 19, 20, 21
    features = np.append(features, crate_diff(self)) # 5 Werte, Indizes 22, 23, 24, 25, 26
    features = np.append(features, no_through_road(self)) # 4 Werte, Indizes 27, 28, 29, 30
    features = np.append(features, no_bomb(self)) # 1 Wert, Index 31
    features = np.append(features, killable_opponents(self)) # 1 Wert, Index 32
    return features



def setup(self):
    self.theta = np.array([[-8.25562019e-01,8.23244276e+00,9.13041752e-01,-1.40443099e+00
,-1.21122335e+00,-1.61056462e+00,7.88551801e-02,-5.29676254e+00
,5.16721382e-01,2.60255119e+00,2.99559787e+00,-5.76688528e-01
,-3.48794838e+00,-1.96409228e+00,-2.82856371e+01,-2.77575383e+00
,-5.42082998e-01,-5.97214280e-01,-6.16120421e+00,1.12388390e+00
,1.24584167e+00,7.62161913e-01,6.54899894e-01,-1.00249770e-01
,3.60003372e-01,-2.63478595e-01,-9.07208038e-01,-1.92909000e+01
,8.67661356e-01,-3.52577182e-01,-5.33318004e-01,2.72341354e-01
,-4.16570405e-01,1.16978411e+00]
,[-6.28104872e-01,-7.79615100e+00,-1.64902261e+00,5.92340676e-01
,1.84298590e+00,-1.39301466e+00,-1.66397287e-01,-5.84494380e-01
,-5.36540474e+00,2.23330846e+00,2.52254928e+00,2.07337495e+00
,-5.25751434e+00,-1.84863683e+00,-2.86062485e+00,-2.64301654e+01
,-1.40784799e-01,-4.92463809e-01,1.42154494e+00,-5.58817200e+00
,8.32594323e-01,5.33974206e-01,-9.46900343e-01,-1.31661527e-01
,-3.64070070e-01,-2.91001053e-01,-9.69887429e-01,1.27618141e+00
,-1.97850711e+01,-1.29493499e-01,-5.54238784e-01,4.17945896e-02
,-4.03856705e-01,1.62812939e+00]
,[-1.11226427e+00,-5.13928611e-01,5.18081608e+00,3.05825820e-01
,2.96300956e+00,5.31649494e-01,-3.29995459e+00,2.21819951e+00
,2.37513074e+00,-5.45135869e+00,4.07037790e-01,-7.72876698e-02
,-4.47216919e+00,-1.84033194e+00,-6.61577766e-01,-2.71637713e-01
,-2.90712993e+01,-2.46604805e+00,1.54343472e+00,6.71301557e-01
,-5.88822882e+00,1.17267416e+00,-1.21563713e-01,6.00870054e-01
,-4.85185273e-02,3.02480401e-01,-7.23788051e-01,-5.40089619e-01
,-7.06649665e-01,-1.96338913e+01,1.01745385e+00,2.66826819e-01
,-1.29695202e-01,2.44849839e+00]
,[-3.31319566e-01,1.50706790e-01,-4.28199940e+00,-1.09957092e-01
,-1.17970232e+00,9.65454947e-02,-3.74171227e+00,1.91963412e+00
,2.34972839e+00,-1.18302119e-01,-5.29862900e+00,1.57609237e+00
,-4.46346757e+00,-1.73678274e+00,-5.23871939e-01,-3.92678736e-02
,-2.97509204e+00,-2.72650059e+01,6.14656289e-01,5.50627383e-01
,1.31693559e+00,-7.06687450e+00,-5.80657276e-02,-9.46186947e-01
,-3.33787914e-02,-4.82020411e-01,-1.09432193e+00,-1.74862796e-01
,-7.37058120e-01,1.19502514e+00,-2.17016734e+01,-4.25208902e-01
,-1.41639607e-01,3.80208263e+00]
,[-1.14484055e-01,-2.11331993e+00,2.88677207e+00,2.28839805e+00
,-2.61502246e+00,-2.51614396e+00,-1.75332891e+00,2.33081543e-01
,9.45921862e-01,4.31281328e-01,3.36555571e-01,-7.26784320e+01
,1.03952729e+00,2.10427218e-01,-5.23446706e+00,-3.36961509e+00
,-3.97459775e+00,-3.50751706e+00,2.34105133e+00,2.32680241e+00
,2.18611038e+00,1.96082619e+00,2.30983671e-01,6.42522814e-02
,2.03412023e-01,-1.06750770e-01,2.13969275e+00,2.39586412e+00
,2.99916429e-01,-8.41383942e-01,1.60741722e+01,-6.28711611e+00
,2.07598962e-01,7.71195933e-02]
,[-3.67599271e-02,-2.86306486e-01,-1.74031887e+00,6.55897732e-01
,1.89084460e+00,-1.71880570e+00,-1.52520312e+00,3.13654211e-01
,-1.48699591e-01,-4.37400488e-01,1.48322761e-01,-1.50885264e+01
,-5.25133479e+01,3.24103523e+00,-1.30348673e+00,-1.26764874e+00
,-5.63062453e-01,-1.84180527e+00,2.59161391e-01,1.17698674e-01
,9.44928689e-02,4.50513344e-02,1.33542142e-01,-3.39240935e-01
,-1.76719955e-01,-5.82077007e-02,2.13302464e+00,-7.89918842e+01
,-3.37058180e+01,2.00000000e+01,-6.08044465e+00,-3.11397239e+01
,5.77459060e+00,-9.07343281e-01]])
    #self.theta = np.load('{}thetas/theta_q.npy'.format(path))
    self.q_data = np.load('{}q_data/q_data.npy'.format(path))
    self.all_data = np.load('{}all_data/all_data.npy'.format(path))



def act(self):
    self.next_action = ['LEFT', 'RIGHT', 'UP', 'DOWN', 'WAIT', 'BOMB'][int(np.argmax(q_function(self.theta, build_features(self))))]

    all_scores = np.loadtxt('score.txt')
    if self.game_state['step'] == 1:
        all_scores = np.append(all_scores, np.zeros((1, 4)), axis = 0)
    score = all_scores[-1]

    own_score = self.game_state['self'][-1]
    score[0] = own_score

    opponents = {'simple_agent_0':1, 'simple_agent_1':2, 'simple_agent_2':3}
    for opp in self.game_state['others']:
         score[opponents[opp[2]]] = opp[-1]

    all_scores[-1] = score
    np.savetxt('score.txt', all_scores, fmt = '%i')

    global scores
    scores = np.reshape(score, (1, 4))

    self.q_data = np.append(self.q_data, [np.array([chosen_action, q_value[chosen_action], *features])], axis=0)
    self.all_data = np.append(self.all_data, [np.array([chosen_action, q_value[chosen_action], *features])], axis=0)

    return None        

def reward_update(self):
    return None
    global moves, round_number
    last_event = self.q_data[-1]
    f = last_event[2:]
    
    #sleep(10)

    event_dict = {0 : "MOVED_LEFT",
            1 : "MOVED_RIGHT",
            2 : "MOVED_UP",
            3 : "MOVED_DOWN",
            4 : "WAITED",
            5 : "INTERRUPTED",
            6 : "INVALID_ACTION",
            7 : "BOMB_DROPPED",
            8 : "BOMB_EXPLODED",
            9 : "CRATE_DESTROYED",
            10 : "COIN_FOUND",
            11 : "COIN_COLLECTED",
            12 : "KILLED_OPPONENT",
            13 : "KILLED_SELF",
            14 : "GOT_KILLED",
            15 : "OPPONENT_ELIMINATED",
            16 : "SURVIVED_ROUND"}

    rewards = {"MOVED_LEFT" : -2 + f[1] + 0.2*f[3] + 0.2*f[5] - 3*f[7] - 35*f[18] + 2*f[22] + 0.4*f[24] - 22*f[27], 
                    "MOVED_RIGHT" : -2 - f[1] - 0.2*f[3] + 0.2*f[5] - 3*f[8] - 35*f[19] - 2*f[22] - 0.4*f[24] - 22*f[28],
                    "MOVED_UP" : -2 + f[2] + 0.2*f[4] + 0.2*f[6] - 3*f[9] - 35*f[20] + 2*f[23] + 0.4*f[25] - 22*f[29],
                    "MOVED_DOWN" : -2 - f[2] - 0.2*f[4] + 0.2*f[6] - 3*f[10] - 35*f[21] - 2*f[23] - 0.4*f[25] - 22*f[30],
                    "WAITED" : -6 - 300*f[11], 
                    "INTERRUPTED" : 0,
                    "INVALID_ACTION" : -5 - 5*f[5] - 5*f[6] - 50*f[12] - 10*np.clip(np.sum(f[14:18]), 0, 1),
                    
                    "BOMB_DROPPED" :  -5 + 5*f[13] + 2*f[26] - 30*f[31] + 10*f[32] - 4*f[11],
                    "BOMB_EXPLODED" : 0, 
                    
                    "CRATE_DESTROYED" : 2,
                    "COIN_FOUND" : 0,
                    "COIN_COLLECTED" : 20,

                    "KILLED_OPPONENT" : 0,
                    "KILLED_SELF" : 0,

                    "GOT_KILLED" : 0,
                    "OPPONENT_ELIMINATED" : 0,
                    "SURVIVED_ROUND" : 0}

    reward = 0.667761 * np.sum([rewards[event_dict[item]] for item in self.events])

    global crate_counter
    for event in self.events:
        if event == 9:
            crate_counter += 1

    moves = np.append(moves, [last_event[0], reward, *f])
    moves = np.reshape(moves, (int(len(moves.flat)/(2+len(f))), 2+len(f)))
    return None


def end_of_episode(self):
    global scores
    all_scores = np.loadtxt('score_end_of_episode.txt')
    all_scores = np.append(all_scores, scores, axis = 0)
    score = all_scores[-1]

    own_score = self.game_state['self'][-1]
    score[0] = own_score

    opponents = {'simple_agent_0':1, 'simple_agent_1':2, 'simple_agent_2':3}
    for opp in self.game_state['others']:
         score[opponents[opp[2]]] = opp[-1]

    all_scores[-1] = score
    np.savetxt('score_end_of_episode.txt', all_scores, fmt = '%i')

    return None
    try:
        global moves, round_number, path, crate_counter
        alpha = 0.04        
        gamma = 0.4
        n = 7    
        for t in range(len(moves)):
            if (t >= len(moves) - n):
                n = len(moves) - t - 1
            q_next = np.max(q_function(self.theta, moves[t + n, 2:])) 
            y_t = np.sum([gamma**(t_ - t - 1) * moves[t_, 1] for t_ in range(t + 1, t + n + 1)]) + gamma**n * q_next

            q_update = self.q_data[-len(moves) + t, 1] + alpha * (y_t - self.q_data[-len(moves) + t, 1])   
            
            self.q_data[-len(moves) + t, 1] = q_update        

        
        with open('{}moves.txt'.format(path), 'a') as f:
            new_coins = self.game_state['self'][4]
            last_score = self.game_state['self'][4]
            f.write(str(round_number) + ' ' + str(len(moves)) + ' ' + str(new_coins) + ' ' + str(crate_counter) + '\n')
            crate_counter = 0

        if (round_number % 100 == 0):
            try: 
                theta = []
                for i in range(6):
                    regression_data = copy.deepcopy(self.q_data)
                    if len(self.all_data) > 6000:
                        tmp = self.all_data[:-6000]
                        if len(tmp) > 6000:
                            tmp = tmp[np.array(random.sample(list(np.arange(len(tmp))), 6000))]
                        regression_data = np.append(tmp, regression_data, axis = 0)
                    
                    mask = regression_data[:,0]==i
                    regression_data = regression_data[mask][:, 1:]
                    
                    popt, pcov = curve_fit(func_curve, (regression_data[:, 1:].T), regression_data[:, 0], p0=self.theta[i])
                    theta.append(popt)
                #self.theta = np.clip(np.array(theta), -90, 20)

            except Exception as e:
                pass
                #print(e)
                #print('theta unverändert')
                #print('Exception as e:')
                #print('line: ', sys.exc_info()[2].tb_lineno)
                #print(type(e), e) 
                #print('Traceback:')
                #traceback.print_tb(sys.exc_info()[2], file = sys.stdout)
                #print('end of traceback.')
                pass
         
        if len(self.q_data) > 6000:
            self.q_data = self.q_data[-6000:]       
        
        if len(self.all_data) > 30000:
            self.all_data = self.all_data[np.array(random.sample(list(np.arange(len(self.all_data))), 30000))]    

        if round_number % 20 == 0:
            #np.save('{}thetas/theta_q.npy'.format(path), self.theta)
            np.save('{}q_data/q_data.npy'.format(path), self.q_data)
            np.save('{}all_data/all_data.npy'.format(path), self.all_data)
            np.save('{}thetas/theta_nach_{}_spielen.npy'.format(path, round_number), self.theta)
            if round_number % 60 == 0:
                np.save('{}all_data/all_data_nach_{}_spielen.npy'.format(path, round_number), self.all_data)

        round_number += 1
        moves = np.array([[]])    
        # print('END')
    except Exception as e:
        pass
        #print('Exception as e:')
        #print('line: ', sys.exc_info()[2].tb_lineno)
        #print(type(e), e) 
        #print('Traceback:')
        #traceback.print_tb(sys.exc_info()[2], file = sys.stdout)
        #print('end of traceback.')
        
    return None


