#! /usr/bin/env python3

from torch._C import ParameterDict
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler

class Configurable_Vanilla(nn.Module):
    def __init__(
            self,
            x_net,
            label_loss_object,
            learning_rate
        ):
        super(Configurable_Vanilla, self).__init__()

        self.label_loss_object = label_loss_object

        self.x_net=x_net


        self.init_weight(self.x_net)


        # self.optimizer_G = torch.optim.Adam(self.netE.parameters(), lr=learning_rate, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
        # self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=learning_rate, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)

        self.non_domain_optimizer = torch.optim.Adam(
            list(self.x_net.parameters()),

            learning_rate
        )

        self.non_domain_scheduler = lr_scheduler.ExponentialLR(optimizer=self.non_domain_optimizer, gamma=0.5 ** (1 / 50))

    def init_weight(self, net=None):
        if net is None:
            net = self
        for m in net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, val=0)


    def forward(self, x):
        x = x.float()

        y_hat = F.log_softmax(self.x_net(x), dim=1)

        return y_hat

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def learn(self, x,y):
        """
        returns a dict of
        {
            label_loss:float
        }
        """

        y_hat = self.forward(x) # Yeah it's dumb but I can't find an easy way to train the two nets separately without this

        label_loss = self.label_loss_object(y_hat, y)
        # END TODO

        self.non_domain_optimizer.zero_grad()
        label_loss.backward()
        self.non_domain_optimizer.step()

        return {
            "label_loss": label_loss
        }
