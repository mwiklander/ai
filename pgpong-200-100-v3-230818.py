""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
# run in cd /Users/magnus/Clo*/mom*/mag*/cod*/ai/pgpong
# Log of runs with this version
# r1 = first run on MacbookPro, didn't have ReLU for H1
# r2 = second run, om Macbook12, not correctly setup up back propagation but did some nice learning anyhow
# r3 = second run on Macbook12, still ot correctly
# r3v2 = continued from the 3000 episodes of run 2, the "running mean" was 19.6 when interrupted just after 3000 eps
# r3v3 = continued after accidentally stopped (resume r3v2)
# New test Aug 2023

import numpy as np
import pickle
import gym

# hyperparameters
H1 = 200 # number of hidden layer neurons, j below
H2 = 100 # number of hidden second layer neurons, k below
batch_size = 10 # every how many episodes to do a param update?
learning_rate = 1e-4
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
resume = True # resume from previous checkpoint?
render = False

# model initialization
D = 80 * 80 # input dimensionality: 80x80 grid, i below
if resume:
  model = pickle.load(open('save200-100-r3v2.p', 'rb'))
else:
  model = {}
  model['W1'] = np.random.randn(H1,D) / np.sqrt(D) # "Xavier" initialization
  model['W2'] = np.random.randn(H2,H1) / np.sqrt(H1) #
  model['W3'] = np.random.randn(H2) / np.sqrt(H2)

grad_buffer = { k : np.zeros_like(v) for k,v in model.items() } # update buffers that add up gradients over a batch
rmsprop_cache = { k : np.zeros_like(v) for k,v in model.items() } # rmsprop memory

def sigmoid(x):
  return 1.0 / (1.0 + np.exp(-x)) # sigmoid "squashing" function to interval [0,1]

def prepro(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  I = I[35:195] # crop
  I = I[::2,::2,0] # downsample by factor of 2
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  return I.astype(np.float64).ravel()

def discount_rewards(r):
  """ take 1D float array of rewards and compute discounted reward """
  discounted_r = np.zeros_like(r)
  running_add = 0
  for t in reversed(range(0, r.size)):
    if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
    running_add = running_add * gamma + r[t]
    discounted_r[t] = running_add
  return discounted_r

def policy_forward(x):
  h1 = np.dot(model['W1'], x)
  h1[h1<0] = 0 # ReLU nonlinearity

  h2 = np.dot(model['W2'],h1)
  h2[h2<0] = 0 # ReLU nonlinearity

  logp = np.dot(model['W3'], h2)
  p = sigmoid(logp)

  return p, h1, h2 # return probability of taking action 2, and hidden state

def policy_backward(eph1, eph2, epx, epdlogp): # Dim eph1: N x 200, eph2: N x 100, epx: N x 80*80, epdlogp: N x 1
  """ backward pass. (eph is array of intermediate hidden states) """
  # Note on domensions below: (dot) eliminates closest indices, so q x w (dot) e x r has dimension q x r
  dW3 = np.dot(eph2.T, epdlogp).ravel() # Dim (N x k)T (dot) N x 1 = k x N (dot) N x 1 --> k x 1

  dh2 = np.outer(epdlogp, model['W3']) # Dim N x k (N number of samples, k number of hidden neurons layer 2)
  # More dim: N x 1 (outer) 100 x 1 --> N x 100 (N x k)
  dh2[eph2 <= 0] = 0 # backpro prelu
  dW2 = np.dot(dh2.T, eph1) # (N x k)T (dot) N x j = k x N (dot) N x j --> k x j

  dh1 = np.dot(model['W2'].T, dh2.T) # Dim (k x j)T (dot) (N x k)T = j x k (dot) k x N --> j x N
  # dh1 = W2 (outer) W3 with ReLU adjustment for H2, ie if H2(k) = 0 then no effect of the gradioent, and consequently that component of the gradient is zero
  dh1[eph1.T <= 0] = 0 # backpro prelu, if H1(j) = 0 then corresponding gradient is zero
  dW1 = np.dot(dh1, epx) # Dim j x N (dot) N x i --> j x i

  return {'W1':dW1, 'W2':dW2, 'W3':dW3}

env = gym.make("Pong-v0", render_mode='human')
observation = env.reset()
observation = observation[0] # Added 230817 because something had changed in OpenAI Gym so that the reset frame is now a tuple
prev_x = None # used in computing the difference frame
xs,h1s,h2s,dlogps,drs = [],[],[],[],[]
running_reward = None
reward_sum = 0
episode_number = 0
while True:
  if render: env.render()

  # preprocess the observation, set input to network to be difference image
  cur_x = prepro(observation)
  x = cur_x - prev_x if prev_x is not None else np.zeros(D)
  prev_x = cur_x

  # forward the policy network and sample an action from the returned probability
  aprob, h1, h2 = policy_forward(x)
  action = 2 if np.random.uniform() < aprob else 3 # roll the dice!

  # record various intermediates (needed later for backprop)
  xs.append(x) # observation
  h1s.append(h1) # hidden state
  h2s.append(h2) # hidden state

  y = 1 if action == 2 else 0 # a "fake label"
  dlogps.append(y - aprob) # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)

  # step the environment and get new measurements
  observation, reward, done, truncated, info = env.step(action)
  reward_sum += reward

  drs.append(reward) # record reward (has to be done after we call step() to get reward for previous action)

  if done: # an episode finished
    episode_number += 1

    # stack together all inputs, hidden states, action gradients, and rewards for this episode
    epx = np.vstack(xs)
    eph1 = np.vstack(h1s)
    eph2 = np.vstack(h2s)
    epdlogp = np.vstack(dlogps)
    epr = np.vstack(drs)
    xs,h1s,h2s,dlogps,drs = [],[],[],[],[] # reset array memory

    # compute the discounted reward backwards through time
    discounted_epr = discount_rewards(epr)
    # standardize the rewards to be unit normal (helps control the gradient estimator variance)
    discounted_epr -= np.mean(discounted_epr)
    discounted_epr /= np.std(discounted_epr)

    epdlogp *= discounted_epr # modulate the gradient with advantage (PG magic happens right here.)
    grad = policy_backward(eph1, eph2, epx, epdlogp)
    for k in model: grad_buffer[k] += grad[k] # accumulate grad over batch

    # perform rmsprop parameter update every batch_size episodes
    if episode_number % batch_size == 0:
      for k,v in model.items():
        g = grad_buffer[k] # gradient
        rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
        model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
        grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer

    # boring book-keeping
    running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
    print ('ep: %d, resetting env. episode reward total was %f. running mean: %f' % (episode_number, reward_sum, running_reward))
    if episode_number % 100 == 0: pickle.dump(model, open('save200-100-r3v3-230818.p', 'wb'))
    reward_sum = 0
    observation = env.reset() # reset env
    observation = observation[0]  # Added 230817 because something had changed in OpenAI Gym so that the reset frame is now a tuple
    prev_x = None

  #if reward != 0: # Pong has either +1 or -1 reward exactly when game ends.
    #print ('ep %d: game finished, reward: %f' % (episode_number, reward))
    #print ('' if reward == -1 else ' !!!!!!!!')
