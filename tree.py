import time
import copy
import numpy as np
from node import Node


# Monte Carlo Tree
class Tree:
    def __init__(self, net):
        self.net = net
        self.nodes = {}

    def search(self, state, path, rp, depth):
        # Return predicted value from new state
        key = state.record_string()
        if len(path) > 0:
            key += '|' + ' '.join(map(state.action2str, path))
        if key not in self.nodes:
            p, v = self.net.prediction.inference(rp)
            self.nodes[key] = Node(p, v)
            return v

        # State transition by an action selected from bandit
        node = self.nodes[key]
        p = node.p
        mask = np.zeros_like(p)
        if depth == 0:
            # Add noise to policy on the root node
            p = 0.75 * p + 0.25 * np.random.dirichlet([0.15] * len(p))
            # On the root node, we choose action only from legal actions
            mask[state.legal_actions()] = 1
            p *= mask
            p /= p.sum() + 1e-16

        n, q_sum = 1 + node.n, node.q_sum_all / node.n_all + node.q_sum
        ucb = q_sum / n + 2.0 * np.sqrt(node.n_all) * p / n + mask * 4 # PUCB formula
        best_action = np.argmax(ucb)

        # Search next state by recursively calling this function
        rp_next = self.net.dynamics.inference(rp, state.action_feature(best_action))
        path.append(best_action)
        q_new = -self.search(state, path, rp_next, depth + 1) # With the assumption of changing player by turn
        node.update(best_action, q_new)

        return q_new

    def think(self, state, num_simulations, temperature = 0, show=False):
        # End point of MCTS
        if show:
            print(state)
        start, prev_time = time.time(), 0
        for _ in range(num_simulations):
            self.search(state, [], self.net.representation.inference(state.feature()), depth=0)

            # Display search result on every second
            if show:
                tmp_time = time.time() - start
                if int(tmp_time) > int(prev_time):
                    prev_time = tmp_time
                    root, pv = self.nodes[state.record_string()], self.pv(state)
                    print('%.2f sec. best %s. q = %.4f. n = %d / %d. pv = %s'
                          % (tmp_time, state.action2str(pv[0]), root.q_sum[pv[0]] / root.n[pv[0]],
                             root.n[pv[0]], root.n_all, ' '.join([state.action2str(a) for a in pv])))

        #  Return probability distribution weighted by the number of simulations
        root = self.nodes[state.record_string()]
        n = root.n + 1
        n = (n / np.max(n)) ** (1 / (temperature + 1e-8))
        return n / n.sum()

    def pv(self, state):
        # Return principal variation (action sequence which is considered as the best)
        s, pv_seq = copy.deepcopy(state), []
        while True:
            key = s.record_string()
            if key not in self.nodes or self.nodes[key].n.sum() == 0:
                break
            best_action = sorted([(a, self.nodes[key].n[a]) for a in s.legal_actions()], key=lambda x: -x[1])[0][0]
            pv_seq.append(best_action)
            s.play(best_action)
        return pv_seq
    