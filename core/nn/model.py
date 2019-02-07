import torch
import torch.nn as nn


class SamplingLayer(nn.Module):
    def __init__(self, n_topics):
        super(SamplingLayer, self).__init__()
        self.noise = torch.autograd.Variable(torch.zeros(n_topics).cuda())

    def forward(self, mu, log_var):
        epsilon = self.noise.data.normal_(0, 1.0)
        return mu + torch.exp(log_var / 2) * epsilon


class NeuralTopicModel(nn.Module):
    def __init__(self,
                 vocab_size,
                 embed_size,
                 hidden_size,
                 n_topics,
                 shortcut=True):
        self.embedding = nn.Embedding(vocab_size, embed_size, sparse=True)
        self.relu = nn.ReLU()
        self.enc1 = nn.Linear(embed_size, hidden_size)
        self.enc2 = nn.Linear(hidden_size, n_topics)
        self.enc3 = nn.Linear(hidden_size, n_topics)

        self.shortcut = shortcut
        if self.shortcut:
            self.enc_short = nn.Embedding(vocab_size, hidden_size, sparse=True)

        self.sampling = SamplingLayer(n_topics)

        self.gen1 = nn.Linear(n_topics, n_topics)
        self.gen2 = nn.Linear(n_topics, n_topics)
        self.gen3 = nn.Linear(n_topics, n_topics)
        self.gen4 = nn.Linear(n_topics, n_topics)
        self.tanh = nn.Tanh()

        self.dec1 = nn.Linear(n_topics, vocab_size)

    def forward(self, bow_input):
        h_linear = self.relu(self.embedding(bow_input))
        h_lin2 = self.relu(self.enc1(h_linear))
        if self.shortcut:
            h_short = self.enc_short(bow_input)
            h_lin2 = torch.add(h_lin2, h_short)
        z_mean = self.enc2(h_lin2)
        z_log_var = self.enc3(h_lin2)
        hidden = self.sampling(z_mean, z_log_var)

        tmp = self.tanh(self.gen1(hidden))
        tmp = self.tanh(self.gen2(tmp))
        tmp = self.tanh(self.gen3(tmp))
        tmp = self.gen4(tmp)
        if self.shortcut:
            represent = torch.add(self.tanh(tmp), hidden)
        else:
            represent = tmp

        tmp = self.tanh(self.gen1(z_mean))
        tmp = self.tanh(self.gen2(tmp))
        tmp = self.tanh(self.gen3(tmp))
        tmp = self.gen4(tmp)
        if self.shortcut:
            represent_mu = torch.add(self.tanh(tmp), z_mean)
        else:
            represent_mu = tmp
        p_x_given_h = self.dec1(represent)
        return represent_mu, p_x_given_h
