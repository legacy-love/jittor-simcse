from jittor.dataset import Dataset

class CLDataset(Dataset):
    def __init__(self, dataset, tokenizer, data_args, training_args):
        super().__init__()
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.mode = training_args.mode
        self.max_seq_len = data_args.max_seq_len
        self.sent0 = []
        self.sent1 = []
        self.sent2 = []
        if self.mode == "unsupervised":
            for entry in self.dataset:
                self.sent0.append(entry["text"])
                self.sent1.append(entry["text"])
        elif self.mode == "supervised":
            for entry in self.dataset:
                self.sent0.append(entry["sent0"])
                self.sent1.append(entry["sent1"])
                self.sent2.append(entry["hard_neg"])

        else:
            raise ValueError(f"mode must be unsupervised or supervised")
            
    def __len__(self):
        # jittor的Dataset的len与pytorch不一样，这里要手动除以batch_size否则dataloader的len和dataset的len相同
        return len(self.dataset)
        # if self.drop_last:
        #     return len(self.dataset) // self.batch_size
        # return (len(self.dataset) - 1) // self.batch_size + 1

    def __getitem__(self, index):
        if self.mode == "unsupervised":
            t0 = self.tokenizer(
                self.sent0[index],
                max_length = self.max_seq_len,
                truncation=True,
                padding="max_length",
                return_tensors="np"
            )
            t1 = self.tokenizer(
                self.sent1[index],
                max_length = self.max_seq_len,
                truncation=True,
                padding="max_length",
                return_tensors="np"
            )
            t2 = t0
            return t0, t1, t2
        elif self.mode == "supervised":
            t0 = self.tokenizer(
                self.sent0[index],
                max_length = self.max_seq_len,
                truncation=True,
                padding="max_length",
                return_tensors="np"
            )
            t1 = self.tokenizer(
                self.sent1[index],
                max_length = self.max_seq_len,
                truncation=True,
                padding="max_length",
                return_tensors="np"
            )
            t2 = self.tokenizer(
                self.sent2[index],
                max_length = self.max_seq_len,
                truncation=True,
                padding="max_length",
                return_tensors="np"
            )
            return t0, t1, t2
        else:
            raise ValueError(f"mode must be unsupervised or supervised")