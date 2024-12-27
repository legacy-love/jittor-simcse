from jittor import Dataset

class CLDataset(Dataset):
    def __init__(self, dataset, tokenizer, data_args, training_args):
        super().__init__()
        self.dataset = dataset["train"]
        self.tokenizer = tokenizer
        self.mode = training_args.mode
        self.max_seq_len = data_args.max_seq_len
        self.sent0 = []
        self.sent1 = []
        self.sent2 = []
        if self.mode == "unsupervised":
            for entry in self.dataset:
                t = self.tokenizer(
                    entry["text"],
                    max_length = self.max_seq_len,
                    truncation=True,
                    padding="max_length",
                    return_tensors="np"
                )
                self.sent0.append(t)
        elif self.mode == "supervised":
            for entry in self.dataset:
                t0 = self.tokenizer(
                    entry["sent0"],
                    max_length = self.max_seq_len,
                    truncation=True,
                    padding="max_length",
                    return_tensors="np"
                )
                t1 = self.tokenizer(
                    entry["sent1"],
                    max_length = self.max_seq_len,
                    truncation=True,
                    padding="max_length",
                    return_tensors="np"
                )
                t2 = self.tokenizer(
                    entry["hard_neg"],
                    max_length = self.max_seq_len,
                    truncation=True,
                    padding="max_length",
                    return_tensors="np"
                )
                self.sent0.append(t0)
                self.sent1.append(t1)
                self.sent2.append(t2)
        else:
            raise ValueError(f"mode must be unsupervised or supervised")
            
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if self.mode == "unsupervised":
            return self.sent0[index], self.sent0[index]
        elif self.mode == "supervised":
            return self.sent0[index], self.sent1[index], self.sent2[index]
        else:
            raise ValueError(f"mode must be unsupervised or supervised")