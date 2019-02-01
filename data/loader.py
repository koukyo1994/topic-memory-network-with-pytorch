import gensim

from pathlib import Path


class DataLoader:
    def __init__(self, filename):
        path = Path(filename)
        assert path.exists()
        assert path.is_file()

        with open(path) as f:
            self.text = gensim.utils.to_unicode(f.read(),
                                                "latin1").strip().split("\n")
        msgs = []
        labels = []
        label_dict = {}

        for i, line in enumerate(self.text):
            msg, label = line.strip().split("######")
            msg = list(gensim.utils.tokenize(msg, lower=True))
            msgs.append(msg)
            if label not in label_dict:
                label_dict[label] = len(label_dict)
            labels.append(label_dict[label])

        self.msgs = msgs
        self.labels = labels
        self.label_dict = label_dict

    def get_data(self):
        return self.msgs, self.labels, self.label_dict
