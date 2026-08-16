# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for Denoising Diffusion GAN. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

'''
Codes adapted from https://github.com/NVlabs/LSGM/blob/main/util/ema.py
'''
import warnings

import torch
from torch.optim import Optimizer


class EMA(Optimizer):
    def __init__(self, opt, ema_decay):
        self.ema_decay = ema_decay
        self.apply_ema = self.ema_decay > 0.
        self.optimizer = opt
        self.state = opt.state
        self.param_groups = opt.param_groups

    def __getattr__(self, name):
        """Delegate anything this proxy does not define to the wrapped optimizer.

        WHY. __init__ above deliberately does NOT call Optimizer.__init__: this
        class is a thin proxy and needs `state`/`param_groups` to stay ALIASED to
        the wrapped optimizer's, which Optimizer.__init__ would replace with fresh
        empty objects. That was harmless on the PyTorch this was written for, but
        torch >= 2.0 moved hook registries into Optimizer.__init__ and the base
        methods now iterate them:

            Optimizer.state_dict()  ->  self._optimizer_state_dict_pre_hooks
            AttributeError: 'EMA' object has no attribute
                            '_optimizer_state_dict_pre_hooks'

        The first state_dict() call is the `Saving content.` at the END of an
        epoch, so an otherwise healthy run trains a complete epoch and then dies
        at the checkpoint — and with --resume it restarts at epoch 0 and does it
        again, indefinitely, while looking exactly like a running job.

        Delegating rather than hand-copying the missing names keeps this working
        across torch versions: the wrapped optimizer is a real, fully initialised
        Optimizer, so it has whatever the base class currently expects. Sharing
        its (empty) hook registries is also semantically right — a hook
        registered on the EMA wrapper should fire for the optimizer it wraps.
        """
        # __getattr__ runs only on FAILED normal lookup, so guard the one
        # attribute it depends on: during __init__, before self.optimizer is set,
        # any miss would otherwise recurse forever.
        if name == 'optimizer':
            raise AttributeError(name)
        return getattr(self.optimizer, name)

    def step(self, *args, **kwargs):
        retval = self.optimizer.step(*args, **kwargs)

        # stop here if we are not applying EMA
        if not self.apply_ema:
            return retval

        ema, params = {}, {}
        for group in self.optimizer.param_groups:
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                state = self.optimizer.state[p]

                # State initialization
                if 'ema' not in state:
                    state['ema'] = p.data.clone()

                if p.shape not in params:
                    params[p.shape] = {'idx': 0, 'data': []}
                    ema[p.shape] = []

                params[p.shape]['data'].append(p.data)
                ema[p.shape].append(state['ema'])

            for i in params:
                params[i]['data'] = torch.stack(params[i]['data'], dim=0)
                ema[i] = torch.stack(ema[i], dim=0)
                ema[i].mul_(self.ema_decay).add_(params[i]['data'], alpha=1. - self.ema_decay)

            for p in group['params']:
                if p.grad is None:
                    continue
                idx = params[p.shape]['idx']
                self.optimizer.state[p]['ema'] = ema[p.shape][idx, :]
                params[p.shape]['idx'] += 1

        return retval

    def load_state_dict(self, state_dict):
        super(EMA, self).load_state_dict(state_dict)
        # load_state_dict loads the data to self.state and self.param_groups. We need to pass this data to
        # the underlying optimizer too.
        self.optimizer.state = self.state
        self.optimizer.param_groups = self.param_groups

    def swap_parameters_with_ema(self, store_params_in_ema):
        """ This function swaps parameters with their ema values. It records original parameters in the ema
        parameters, if store_params_in_ema is true."""

        # stop here if we are not applying EMA
        if not self.apply_ema:
            warnings.warn('swap_parameters_with_ema was called when there is no EMA weights.')
            return

        for group in self.optimizer.param_groups:
            for i, p in enumerate(group['params']):
                if not p.requires_grad:
                    continue
                ema = self.optimizer.state[p]['ema']
                if store_params_in_ema:
                    tmp = p.data.detach()
                    p.data = ema.detach()
                    self.optimizer.state[p]['ema'] = tmp
                else:
                    p.data = ema.detach()
