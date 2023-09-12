# Main algorithm of MuZero

from net import Net
from state import State
from tree import Tree
from train import train

import numpy as np
import torch.optim as optim


#  Battle against random agents
def vs_random(net, n=100):
    results = {}
    for i in range(n):
        first_turn = i % 2 == 0
        turn = first_turn
        state = State()
        while not state.terminal():
            if turn:
                p, _ = net.predict(state, [])[-1]
                action = sorted([(a, p[a]) for a in state.legal_actions()], key=lambda x:-x[1])[0][0]
            else:
                action = np.random.choice(state.legal_actions())
            state.play(action)
            turn = not turn
        r = state.terminal_reward() if turn else -state.terminal_reward()
        results[r] = results.get(r, 0) + 1
    return results


num_games = 50000
num_games_one_epoch = 20
num_simulations = 40

net = Net()
optimizer = optim.SGD(net.parameters(), lr=3e-4, weight_decay=3e-5, momentum=0.8)

# Display battle results as {-1: lose 0: draw 1: win}
# (for episode generated for training, 1 means that the first player won)
vs_random_sum = vs_random(net)
print('vs_random = ', sorted(vs_random_sum.items()))

episodes = []
result_distribution = {1: 0, 0: 0, -1: 0}

for g in range(num_games):
    # Generate one episode
    record, p_targets, features, action_features = [], [], [], []
    state = State()
    # temperature using to make policy targets from search results
    temperature = 0.7

    while not state.terminal():
        tree = Tree(net)
        p_target = tree.think(state, num_simulations, temperature)
        p_targets.append(p_target)
        features.append(state.feature())

        # Select action with generated distribution, and then make a transition by that action
        action = np.random.choice(np.arange(len(p_target)), p=p_target)
        record.append(action)
        action_features.append(state.action_feature(action))
        state.play(action)
        temperature *= 0.8

    # reward seen from the first turn player
    reward = state.terminal_reward() * (1 if len(record) % 2 == 0 else -1)
    result_distribution[reward] += 1
    episodes.append((record, reward, features, action_features, p_targets))

    if g % num_games_one_epoch == 0:
        print('game ', end='')
    print(g, ' ', end='')

    # Training of neural net
    if (g + 1) % num_games_one_epoch == 0:
        # Show the result distributiuon of generated episodes
        print('generated = ', sorted(result_distribution.items()))
        net = train(episodes, net, optimizer)
        vs_random_once = vs_random(net)
        print('vs_random = ', sorted(vs_random_once.items()), end='')
        for r, n in vs_random_once.items():
            vs_random_sum[r] += n
        print(' sum = ', sorted(vs_random_sum.items()))
        #show_net(net, State())
        #show_net(net, State().play('A1 C1 A2 C2'))
        #show_net(net, State().play('A1 B2 C3 B3 C1'))
        #show_net(net, State().play('B2 A2 A3 C1 B3'))
        #show_net(net, State().play('B2 A2 A3 C1'))
print('finished')