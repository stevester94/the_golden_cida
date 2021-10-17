#! /usr/bin/env python3

from torch._C import ParameterDict
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler

class Configurable_CIDA(nn.Module):
    def __init__(
            self,
            x_net,
            u_net,
            merge_net,
            class_net,
            domain_net,
            label_loss_object,
            domain_loss_object,
            learning_rate
        ):
        super(Configurable_CIDA, self).__init__()

        self.label_loss_object = label_loss_object
        self.domain_loss_object = domain_loss_object

        self.x_net=x_net
        self.u_net=u_net
        self.merge_net=merge_net
        self.class_net=class_net
        self.domain_net=domain_net

        self.init_weight(self.x_net)
        self.init_weight(self.u_net)
        self.init_weight(self.merge_net)
        self.init_weight(self.class_net)
        self.init_weight(self.domain_net)

        # self.optimizer_G = torch.optim.Adam(self.netE.parameters(), lr=learning_rate, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
        # self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=learning_rate, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)

        self.non_domain_optimizer = torch.optim.Adam(
            list(self.x_net.parameters()) + 
            list(self.u_net.parameters()) + 
            list(self.merge_net.parameters()) + 
            list(self.class_net.parameters()),
            learning_rate
        )

        self.domain_optimizer = torch.optim.Adam(self.domain_net.parameters(), lr=learning_rate)

        self.non_domain_scheduler = lr_scheduler.ExponentialLR(optimizer=self.non_domain_optimizer, gamma=0.5 ** (1 / 50))
        self.domain_scheduler     = lr_scheduler.ExponentialLR(optimizer=self.domain_optimizer, gamma=0.5 ** (1 / 50))

    def init_weight(self, net=None):
        if net is None:
            net = self
        for m in net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, val=0)


    def forward(self, x, u):
        x = x.float()
        u = u.float()

        x_feature = self.x_net(x)
        u_feature = self.u_net(u)

        feature = self.merge_net(torch.cat((x_feature, u_feature), dim=1))

        y_hat = F.log_softmax(self.class_net(feature), dim=1)
        u_hat = self.domain_net(feature)

        return y_hat, u_hat

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def learn(self, x,y,u,s, alpha):
        """
        returns a dict of
        {
            label_loss:float, # if domain_only==False
            domain_loss:float
        }
        """

        y_hat, u_hat = self.forward(x,u)

        # Do domain's loss first, it is straight forward
        domain_loss = self.domain_loss_object(u_hat, u)
        label_loss = self.label_loss_object(y_hat[s==1], y[s==1])

        # Note the negation for domain loss against encoder
        encoder_loss = label_loss
        encoder_loss += - alpha * domain_loss

        # self.set_requires_grad(self.class_net, False)
        # self.set_requires_grad(self.domain_net, True)
        self.non_domain_optimizer.zero_grad()
        self.domain_optimizer.zero_grad()

        self.set_requires_grad(self.domain_net, True)
        self.set_requires_grad(self.x_net, False)
        self.set_requires_grad(self.u_net, False)
        self.set_requires_grad(self.merge_net, False)
        self.set_requires_grad(self.class_net, False)
        domain_loss.backward(retain_graph=True)

        self.set_requires_grad(self.domain_net, False)
        self.set_requires_grad(self.x_net, True)
        self.set_requires_grad(self.u_net, True)
        self.set_requires_grad(self.merge_net, True)
        self.set_requires_grad(self.class_net, True)
        encoder_loss.backward(retain_graph=True)

        self.domain_optimizer.step()
        self.non_domain_optimizer.step()

            # d_dyuh = list(map(lambda p: torch.flatten(p), self.domain_net.parameters()))
            # d_dyuh = torch.cat(d_dyuh)
            # d_dyuh = torch.sum(d_dyuh)

            # x_dyuh = list(map(lambda p: torch.flatten(p), self.x_net.parameters()))
            # x_dyuh = torch.cat(x_dyuh)
            # x_dyuh = torch.sum(x_dyuh)

            # print(d_dyuh, x_dyuh)

        return {
            "domain_loss": domain_loss,
            "label_loss": label_loss
        }
