import torch
import torch.nn as nn
from model import MultiExit
from torch.nn import functional as F

from einops import repeat
import numpy as np


class KulbackLeibler(nn.Module):
    def __init__(self, args):
        super(KulbackLeibler, self).__init__()
        self.args = args


        v = torch.tensor([[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]])
        self.v = repeat(v, 'b c -> (b r) c', r=self.args.keynum).to(self.args.device)

    def kl(self, x, exit_target):
        me1, me2, me3, me4 = x  # Multi-exit 1, ... , 4
        loss = torch.empty(4)

        loss[0] = F.kl_div(
            F.log_softmax(me1, dim=1),
            F.log_softmax(self.v, dim=1),
            reduction='sum',
            log_target=True)

        loss[1] = F.kl_div(
            F.log_softmax(me2, dim=1),
            F.log_softmax(self.v, dim=1),
            reduction='sum',
            log_target=True)

        loss[2] = F.kl_div(
            F.log_softmax(me3, dim=1),
            F.log_softmax(self.v, dim=1),
            reduction='sum',
            log_target=True)

        loss[3] = F.kl_div(
            F.log_softmax(me4, dim=1),
            F.log_softmax(self.v, dim=1),
            reduction='sum',
            log_target=True)

        kl = torch.sum(loss)-loss[exit_target]
        return kl

    def forward(self, yhat, exit_target):

        loss = self.kl(x=yhat, exit_target=exit_target)
        return loss


class Keyprint:
    def __init__(self, args):
        super(Keyprint, self).__init__()
        self.args = args

        self.model = MultiExit(mode='train', threshold=self.args.threshold)
        self.model.to(self.args.device)

        param = torch.load('my_model.pth')
        self.model.load_state_dict(param['model_state'])

        self.model.to(self.args.device)
        self.kl_loss = KulbackLeibler(args=args)

    def generate_keyprint(self, image):
        for ii in range (4):  # multi exit number FIXME
            if ii==0:
                keyprints = self.sd_attack(x=image, exit_target=ii)
            else:
                keyprints = torch.column_stack((keyprints, self.sd_attack(x=image, exit_target=ii)))
        return keyprints

    def sd_attack(self, x, exit_target):
        self.model.mode = 'train'
        x = x.to(self.args.device)
        x.requires_grad = True

        # Generate prediction for all exit gate
        yhat = self.model(x)

        # calculate loss
        loss = self.kl_loss(yhat, exit_target)

        # zero all existing gradients
        self.model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # collect datagrad
        data_grad = x.grad.data

        # Collect the element-wise sign of the data gradient
        sign_data_grad = data_grad.sign()

        # Create the perturbed image by adjusting each pixel of the input image
        perturbed_image = x + self.args.epsilon * sign_data_grad

        # Adding clipping to maintain [0,1] range
        keyprint = torch.clamp(perturbed_image, 0, 1)

        # Return the perturbed image
        return keyprint

    def verification(self, keyprints):
        # Set model to inference mode
        self.model.mode = 'inference'
        logger = np.empty((0,4))
        for i in range(self.args.keynum):
            for ii in range(4):  # number of exit gate
                keyprint = keyprints[i,ii].reshape((1,3,224,224))
                keyprint = keyprint.to(self.args.device)

                # Init loggers
                starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                repetition = 100
                logging = np.zeros((repetition, 1))
                with torch.no_grad():
                    for rep in range(repetition):
                        starter.record()
                        _, gate = self.model(keyprint)
                        ender.record()

                        # Wait for GPU Sync
                        torch.cuda.synchronize()
                        curr_time = starter.elapsed_time(ender)
                        logging[rep] = curr_time
                mean_syn = np.sum(logging) / repetition
                logs = [i, ii, gate, mean_syn]
                logger = np.vstack((logger, logs))
        logger = logger.reshape((self.args.keynum,4, 4))
        return logger

    def evaluation_result(self, logs):
        counter = 0
        for i in range(self.args.keynum):
            snippet = logs[i]
            if snippet[0,1] == snippet[0,2] or snippet[1,1] == snippet[1,2] or snippet[2,1] == snippet[2,2] or snippet[3,1] == snippet[3,2]:
                counter += 1
        return counter/self.args.keynum


if __name__ == '__main__':
    keyprint = Keyprint

    model = MultiExit(mode='inference')
    state_dict = torch.load('my_model.pth')
    model.load_state_dict(state_dict['model_state'])

    inputs = torch.zeros((1, 3, 224, 224))

    o1, o2, o3, o4 = model(inputs)
    print()

