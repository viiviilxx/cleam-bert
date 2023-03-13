import pickle, glob, torch, json
import xml.dom.minidom
from tqdm import tqdm
from collections import defaultdict
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

files = glob.glob("../dataset/rcv1v2/raw/*newsML.xml")

with open("../dataset/rcv1v2/meta_data/rcv1v2-ids.dat", "r", encoding="utf-8_sig") as f:
    used_samples = set([i.strip() for i in f.readlines()])

with open("../dataset/rcv1v2/meta_data/rcv1-v2.topics.qrels", "r", encoding="utf-8_sig") as f:
    document_labels = defaultdict(list)
    for line in f.readlines():
        label, document_id = line.strip().split()[:-1]
        document_labels[document_id].append(label)

with open("../dataset/rcv1v2/meta_data/rcv1v2_train_ids.txt", "r", encoding="utf-8_sig") as f:
    train_set_ids = set([line.strip() for line in f.readlines()])

with open("../dataset/rcv1v2/meta_data/rcv1v2_val_ids.txt", "r", encoding="utf-8_sig") as f:
    val_set_ids = set([line.strip() for line in f.readlines()])

with open("../dataset/rcv1v2/meta_data/rcv1v2_test_ids.txt", "r", encoding="utf-8_sig") as f:
    test_set_ids = set([line.strip() for line in f.readlines()])

with open("../dataset/rcv1v2/meta_data/rcv1v2_label_list.tsv", "r") as f:
    label_list = {line.split("\t")[0].strip(): line.split("\t")[1].strip() for line in f.readlines()}
    label_index = {label: i for i, label in enumerate(label_list.keys())}


def make_label_vector(document_id):
    labels = document_labels[document_id]
    label_vector = torch.zeros((1, 103))
    for label in labels:
        index = label_index[label]
        label_vector[0, index] = 1
    return label_vector


def tokenize_label():
    cls_id = torch.tensor([[101]])
    sep_id = torch.tensor([[102]])
    unmask = torch.ones((1, 1))

    label_input_ids = []
    label_attention_mask = []
    for label, detail in label_list.items():
        label_tokens = tokenizer(
            detail, add_special_tokens=False, max_length=3, padding="max_length", return_tensors="pt"
        )
        label_input_ids.append(label_tokens["input_ids"])
        label_attention_mask.append(label_tokens["attention_mask"])

    label_input_ids = torch.cat(label_input_ids, dim=-1)
    label_input_ids = torch.cat([cls_id, label_input_ids, sep_id], dim=-1)
    label_attention_mask = torch.cat(label_attention_mask, dim=-1)
    label_attention_mask = torch.cat([unmask, label_attention_mask, unmask], dim=-1)

    print("label_length :", label_input_ids.size())

    label_token = {"input_ids": label_input_ids, "attention_mask": label_attention_mask}
    with open("../dataset/rcv1v2/pickle/bert_rcv1v2_labels", "wb") as f:
        pickle.dump(label_token, f)


def tokenize_text():
    train_set = {"input_ids": [], "attention_masks": [], "labels": []}
    val_set = {"input_ids": [], "attention_masks": [], "labels": []}
    test_set = {"input_ids": [], "attention_masks": [], "labels": []}

    for file in tqdm(files):
        dom = xml.dom.minidom.parse(file)
        root = dom.documentElement
        document_id = root.getAttributeNode("itemid").nodeValue

        if not document_id in used_samples:
            continue

        label_vector = make_label_vector(document_id)

        tags = root.getElementsByTagName("p")
        text = ""
        for tag in tags:
            text += tag.firstChild.data

        if text == "":
            continue

        text_tokens = tokenizer(text, max_length=512, truncation=True, padding="max_length", return_tensors="pt")

        if document_id in train_set_ids:
            train_set["input_ids"].append(text_tokens["input_ids"])
            train_set["attention_masks"].append(text_tokens["attention_mask"])
            train_set["labels"].append(label_vector)
        elif document_id in val_set_ids:
            val_set["input_ids"].append(text_tokens["input_ids"])
            val_set["attention_masks"].append(text_tokens["attention_mask"])
            val_set["labels"].append(label_vector)
        elif document_id in test_set_ids:
            test_set["input_ids"].append(text_tokens["input_ids"])
            test_set["attention_masks"].append(text_tokens["attention_mask"])
            test_set["labels"].append(label_vector)

    train_set["input_ids"] = torch.cat(train_set["input_ids"], dim=0).to(torch.long)
    train_set["attention_masks"] = torch.cat(train_set["attention_masks"], dim=0).to(torch.long)
    train_set["labels"] = torch.cat(train_set["labels"], dim=0)

    val_set["input_ids"] = torch.cat(val_set["input_ids"], dim=0).to(torch.long)
    val_set["attention_masks"] = torch.cat(val_set["attention_masks"], dim=0).to(torch.long)
    val_set["labels"] = torch.cat(val_set["labels"], dim=0)

    test_set["input_ids"] = torch.cat(test_set["input_ids"], dim=0).to(torch.long)
    test_set["attention_masks"] = torch.cat(test_set["attention_masks"], dim=0).to(torch.long)
    test_set["labels"] = torch.cat(test_set["labels"], dim=0)

    print("train : ", train_set["input_ids"].size())
    print("val : ", val_set["input_ids"].size())
    print("test : ", test_set["input_ids"].size())

    with open("../dataset/rcv1v2/pickle/bert_rcv1v2_train", "wb") as f:
        pickle.dump(train_set, f)
    with open("../dataset/rcv1v2/pickle/bert_rcv1v2_val", "wb") as f:
        pickle.dump(val_set, f)
    with open("../dataset/rcv1v2/pickle/bert_rcv1v2_test", "wb") as f:
        pickle.dump(test_set, f)


def main():
    tokenize_label()
    tokenize_text()


if __name__ == "__main__":
    main()
