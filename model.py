import torch
import torch.nn as nn 
import torch.nn.functional as F  
import pyro
from pyro.distributions import Normal, Categorical

class BNN(nn.Module): 
    def __init__(self, input_size, output_size, hidden_size): 
        super().__init__()
        self.fc = nn.Linear(input_size, hidden_size) 
        self.output = nn.Linear(hidden_size, output_size) 
        
    def forward(self, x):
        out = self.fc(x)
        out = F.relu(out)  
        out = self.output(out) 
        return out 
    

def model(x_data, y_data, net = BNN(28*28, 10, 1024)):
    ## piror for dense layer 
    fc_weight_prior = Normal(loc=torch.zeros_like(net.fc.weight), scale=torch.ones_like(net.fc.weight))
    fc_bias_prior = Normal(loc=torch.zeros_like(net.fc.bias), scale=torch.ones_like(net.fc.bias))
    ## prior for output layer 
    output_weight_prior = Normal(loc=torch.zeros_like(net.output.weight), scale=torch.ones_like(net.output.weight))
    output_bias_prior = Normal(loc=torch.zeros_like(net.output.bias), scale=torch.ones_like(net.output.bias))
    ## storing the priors  
    priors = {'fc.weight': fc_weight_prior, 
              'fc.bias': fc_bias_prior,  
              'output.weight': output_weight_prior, 
              'output.bias': output_bias_prior}
    # lift module parameters to random variables sampled from the priors
    lifted_module = pyro.random_module("module", net, priors)
    # sample a regressor (which also samples w and b)
    lifted_reg_model = lifted_module()
    log_softmax = nn.LogSoftmax(dim=1) 
    lhat = log_softmax(lifted_reg_model(x_data))
    pyro.sample("obs", Categorical(logits=lhat), obs=y_data)


def guide(x_data, y_data, net = BNN(28*28, 10, 1024)):
    softplus = torch.nn.Softplus()
    # First layer weight distribution priors
    fc_weight_mu = torch.randn_like(net.fc.weight)
    fc_weight_sigma = torch.randn_like(net.fc.weight)
    fc_weight_mu_param = pyro.param("fc_weight_mu", fc_weight_mu)
    fc_weight_sigma_param = softplus(pyro.param("fc_weight_sigma", fc_weight_sigma))
    fc_weight_prior = Normal(loc = fc_weight_mu_param, scale = fc_weight_sigma_param)
    # First layer bias distribution priors
    fc_bias_mu = torch.randn_like(net.fc.bias)
    fc_bias_sigma = torch.randn_like(net.fc.bias)
    fc_bias_mu_param = pyro.param("fc_bias_mu", fc_bias_mu)
    fc_bias_sigma_param = softplus(pyro.param("fc_bias_sigma", fc_bias_sigma))
    fc_bias_prior = Normal(loc=fc_bias_mu_param, scale = fc_bias_sigma_param)
    # Output layer weight distribution priors
    output_weight_mu = torch.randn_like(net.output.weight)
    output_weight_sigma = torch.randn_like(net.output.weight)
    output_weight_mu_param = pyro.param("output_weight_mu", output_weight_mu)
    output_weight_sigma_param = softplus(pyro.param("output_weight_sigma", output_weight_sigma))
    output_weight_prior = Normal(loc = output_weight_mu_param, 
                        scale=output_weight_sigma_param).independent(1)
    # Output layer bias distribution priors
    output_bias_mu = torch.randn_like(net.output.bias)
    output_bias_sigma = torch.randn_like(net.output.bias)
    output_bias_mu_param = pyro.param("output_bias_mu", output_bias_mu)
    output_bias_sigma_param = softplus(pyro.param("output_bias_sigma", output_bias_sigma))
    output_bias_prior = Normal(loc = output_bias_mu_param, scale=output_bias_sigma_param)

    priors = {'fc.weight': fc_weight_prior, 'fc.bias': fc_bias_prior, 'output.weight': output_weight_prior, 'output.bias': output_bias_prior}
    lifted_module = pyro.random_module("module", net, priors)
    return lifted_module()
