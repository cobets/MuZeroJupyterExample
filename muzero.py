# Main algorithm of MuZero

import torch
import torch.optim as optim
import numpy as np

from net import Net
from dotsenv import DotsEnv
from tree import Tree
from train import train
import config as cfg


#  Battle against random agents
def vs_random(net, state_class, n=100):
    results = {}
    for i in range(n):
        first_turn = i % 2 == 0
        turn = first_turn
        state = state_class()
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


def muzero(state_class):
    net = Net(state_class)
    optimizer = optim.SGD(net.parameters(), lr=3e-4, weight_decay=3e-5, momentum=0.8)

    # Display battle results as {-1: lose 0: draw 1: win}
    # (for episode generated for training, 1 means that the first player won)
    vs_random_sum = vs_random(net, state_class)
    print('vs_random = ', sorted(vs_random_sum.items()))

    episodes = []
    result_distribution = {1: 0, 0: 0, -1: 0}

    for g in range(cfg.num_games):
        # Generate one episode
        record, p_targets, features, action_features = [], [], [], []
        state = state_class()
        # temperature using to make policy targets from search results
        temperature = 0.7

        while not state.terminal():
            tree = Tree(net)
            p_target = tree.think(state, cfg.num_simulations, temperature)
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

        if g % cfg.num_games_one_epoch == 0:
            print(f'game {g} ', end='')
        print('_', end='')

        # Training of neural net
        if (g + 1) % cfg.num_games_one_epoch == 0:
            # Show the result distribution of generated episodes
            print('generated =', sorted(result_distribution.items()), end=' ')
            net = train(episodes, net, optimizer)
            vs_random_once = vs_random(net, state_class)
            print('vs_random =', sorted(vs_random_once.items()), end='')
            for r, n in vs_random_once.items():
                vs_random_sum[r] += n
            print(' sum =', sorted(vs_random_sum.items()))
            #  show_net(net, State())
            #  show_net(net, State().play('A1 C1 A2 C2'))
            #  show_net(net, State().play('A1 B2 C3 B3 C1'))
            #  show_net(net, State().play('B2 A2 A3 C1 B3'))
            #  show_net(net, State().play('B2 A2 A3 C1'))

    print('saving model...')
    torch.save(net.state_dict(), f'muzero-tictactoe-model-{cfg.num_games}.pth')
    print('finished')


if __name__ == '__main__':
    class StateClassSized(DotsEnv):
        def __init__(self):
            super(StateClassSized, self).__init__(8, 8)
            self.record = []

        def play(self, action):
            super().play(action)
            self.record.append(action)

        def record_string(self):
            return ' '.join([str(a) for a in self.record])

        def action2str(self, a):
            return str(a)


    muzero(StateClassSized)
