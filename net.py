import torch.nn as nn
from representation import Representation
from prediction import Prediction
from dynamics import Dynamics
import config as cfg


# Whole net
class Net(nn.Module):
    def __init__(self, state_class):
        super().__init__()
        state = state_class()
        input_shape = state.feature().shape
        action_shape = state.action_feature(0).shape
        rp_shape = (cfg.num_filters, *input_shape[1:])

        self.representation = Representation(input_shape)
        self.prediction = Prediction(action_shape)
        self.dynamics = Dynamics(rp_shape, action_shape)

    # Predict p and v from original state and path
    def predict(self, state0, path):
        outputs = []
        x = state0.feature()
        rp = self.representation.inference(x)
        outputs.append(self.prediction.inference(rp))
        for action in path:
            a = state0.action_feature(action)
            rp = self.dynamics.inference(rp, a)
            outputs.append(self.prediction.inference(rp))
        return outputs

    def show_net(self, state):
        # Display policy (p) and value (v)
        print(state)
        p, v = self.predict(state, [])[-1]
        print('p = ')
        print((p * 1000).astype(int).reshape((-1, *self.representation.input_shape[1:3])))
        print('v = ', v)
        print()
