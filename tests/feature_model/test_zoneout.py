import unittest

import torch

from torch.autograd import Variable

from src.feature_model.zoneout import zoneout
from src.feature_model.zoneout import Zoneout


class TestZoneout(unittest.TestCase):

    def test_Zoneout(self):
        previous_input = torch.randn(5000)
        current_input = torch.randn(5000)
        previous_mean, current_mean = previous_input.mean(), current_input.mean()
        for p in [0, 0.15, 1]:
            module = Zoneout(p)
            module.training = True
            current_input_var = Variable(current_input, requires_grad=True)
            previous_input_var = Variable(previous_input, requires_grad=True)
            output = module(current_input_var, previous_input_var)
            expected_mean = p * current_mean + (1 - p) * previous_mean
            self.assertLess(abs(output.data.mean() - expected_mean), 0.1)
            output.backward(current_input)
            current_input_grad = current_input_var.grad.data
            self.assertLess(abs(current_input_grad.mean() - p * current_input.mean()), 0.1)

            # Check that these don't raise errors
            module.__repr__()
            str(module)

    def test_Zoneout_during_inference(self):
        current_input_var = Variable(torch.randn(5000), requires_grad=True)
        previous_input_var = Variable(torch.randn(5000), requires_grad=True)
        module = Zoneout(0.5)
        module.training = False
        output = module(current_input_var, previous_input_var)
        self.assertTrue(output.equal(current_input_var))
        output.backward(current_input_var.data)
        self.assertTrue(current_input_var.equal(current_input_var.grad))

    def test_Zoneout_with_shared_mask(self):
        previous_input = torch.randn(5000)
        current_input = torch.randn(5000)
        p = 0.2
        probabilities = torch.Tensor(5000).fill_(p)
        mask = torch.ByteTensor(5000)
        mask.bernoulli_(probabilities)
        mask_mean = mask.type(torch.FloatTensor).mean()
        module = Zoneout(mask=mask)
        module.training = True
        current_input_var = Variable(current_input, requires_grad=True)
        previous_input_var = Variable(previous_input, requires_grad=True)
        output = module(current_input_var.clone(), previous_input_var.clone())
        output2 = module(current_input_var.clone(), previous_input_var.clone())
        # make sure mask is shared across time-steps
        self.assertLess(abs(output.data.mean() - output2.data.mean()), 0.1)
        output.backward(current_input.clone())
        current_input_grad = current_input_var.grad.data
        self.assertLess(abs(current_input_grad.mean() - mask_mean * current_input.mean()), 0.1)

    def test_Zoneout_argument_validation(self):
        self.assertRaises(ValueError, lambda: Zoneout())
        self.assertRaises(ValueError, lambda: Zoneout(1.3))
        self.assertRaises(ValueError, lambda: Zoneout(2))
        self.assertRaises(ValueError, lambda: Zoneout(-2))
        self.assertRaises(ValueError, lambda: Zoneout(mask=torch.Tensor(5)))

        v = Variable(torch.ones(1))
        self.assertRaises(ValueError, lambda: zoneout(v, v, p=3.5))
        self.assertRaises(ValueError, lambda: zoneout(v, v, mask=v))
