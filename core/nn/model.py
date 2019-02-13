import torch
import torch.nn as nn


class SamplingLayer(nn.Module):
    def __init__(self, n_topics):
        super(SamplingLayer, self).__init__()
        self.noise = torch.autograd.Variable(torch.zeros(n_topics).cuda())

    def forward(self, mu, log_var):
        epsilon = self.noise.data.normal_(0, 1.0)
        return mu + torch.exp(log_var / 2) * epsilon


class Flatten(nn.Module):
    def forward(self, inp):
        return inp.view(inp.size(0), -1)


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
        return represent_mu, p_x_given_h, z_mean, z_log_var


class TopicMemoryNetwork(nn.Module):
    def __init__(self,
                 vocab_size,
                 max_features,
                 max_len,
                 embed_size,
                 topic_num,
                 topic_emb_dim,
                 n_classes,
                 embedding_matrix,
                 filter_sizes=[1, 2, 3],
                 num_filters=512):
        super(TopicMemoryNetwork, self).__init__()
        self.vocab_size = vocab_size
        self.max_features = max_features
        self.max_len = max_len
        self.embed_size = embed_size
        self.topic_num = topic_num
        self.topic_emb_dim = topic_emb_dim
        self.num_filters = num_filters
        self.n_classes = n_classes

        self.seq_emb = nn.Embedding(max_features, embed_size)
        self.seq_emb.weight = nn.Parameter(
            embedding_matrix, dtype=torch.float32)
        self.seq_emb.weight.requires_grad = False

        self.relu = nn.ReLU()

        self.c1 = nn.Linear(embed_size, topic_emb_dim)
        self.dropout = nn.Dropout(0.5)

        self.topic_emb = nn.Embedding(topic_num, vocab_size)
        self.topic_emb.weight.requires_grad = False

        self.t1 = nn.Linear(vocab_size, topic_emb_dim)
        self.f1 = nn.Linear(topic_num, topic_emb_dim)
        self.o1 = nn.Linear(topic_emb_dim, topic_emb_dim)

        self.conv0 = nn.Conv2d(1, num_filters,
                               (filter_sizes[0], topic_emb_dim))
        self.conv1 = nn.Conv2d(1, num_filters,
                               (filter_sizes[1], topic_emb_dim))
        self.conv2 = nn.Conv2d(1, num_filters,
                               (filter_sizes[2], topic_emb_dim))

        self.maxpool0 = nn.MaxPool2d((max_len + 1 - filter_sizes[0], 1))
        self.maxpool1 = nn.MaxPool2d((max_len + 1 - filter_sizes[1], 1))
        self.maxpool2 = nn.MaxPool2d((max_len + 1 - filter_sizes[2], 1))

        self.flatten = Flatten()
        self.dropout2 = nn.Dropout(0.05)
        self.out = nn.Linear(3 * num_filters, n_classes)

    def forward(self, seq_input, psudo_input, bow_input, ntm_model):
        x = self.seq_emb(seq_input)
        x = self.dropout(self.relu(self.c1(x)))

        wt_emb = self.topic_emb(psudo_input)
        wt_emb = self.relu(self.t1(wt_emb))

        match = torch.bmm(x, wt_emb.transpose(1, 2))
        represent_mu, _, _, _ = ntm_model(bow_input)
        joint_match = torch.add(represent_mu, match)
        joint_match = self.relu(self.f1(joint_match))
        topic_sum = torch.add(joint_match, x)
        topic_sum = self.relu(self.o1(topic_sum)).reshape((1, self.max_len,
                                                           self.topic_emb_dim))

        x0 = self.relu(self.conv0(topic_sum))
        x1 = self.relu(self.conv1(topic_sum))
        x2 = self.relu(self.conv2(topic_sum))

        mp0 = self.maxpool0(x0).reshape((-1, 1, 1, self.num_filters))
        mp1 = self.maxpool1(x1).reshape((-1, 1, 1, self.num_filters))
        mp2 = self.maxpool2(x2).reshape((-1, 1, 1, self.num_filters))

        cat = torch.cat([mp0, mp1, mp2], dim=1)
        flt = self.flatten(cat)
        out = self.out(flt)
        return out
