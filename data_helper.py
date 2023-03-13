import pickle
from torch.utils.data import Dataset


def load_document_data(path):
    print("loading " + path + "... ", end="", flush=True)
    with open(path, "rb") as f:
        document = pickle.load(f)
    print("done!", flush=True)
    return document["input_ids"], document["attention_masks"], document["labels"]


def load_label_tokens(path):
    with open(path, "rb") as f:
        tokens = pickle.load(f)
    return tokens


class BertHelper(Dataset):
    def __init__(self, path, device):
        super(BertHelper, self).__init__()
        self.text_ids, self.text_masks, self.label_vectors = load_document_data(path)
        self.device = device

    def __len__(self):
        return self.text_ids.size()[0]

    def __getitem__(self, index):
        text = self.text_ids[index].to(self.device)
        mask = self.text_masks[index].to(self.device)
        label = self.label_vectors[index].to(self.device)
        return text, mask, label
