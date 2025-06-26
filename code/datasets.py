import os
import os.path as osp
from torch.utils import data


class SourceDataSet(data.Dataset):
    def __init__(self, seqs, motifs, features, expression):
        super(SourceDataSet, self).__init__()
        self.seq = seqs
        self.motif = motifs
        self.fea = features
        self.exp = expression

        assert len(self.seq) == len(self.exp) and len(self.fea) == len(self.exp), \
        "the number of sequences and labels must be consistent."

    def __len__(self):
        return len(self.exp)

    def __getitem__(self, index):
        seq_one = self.seq[index]
        motif_one = self.motif[index]
        fea_one = self.fea[index]
        exp_one = self.exp[index]

        return {"seq": seq_one, "motif": motif_one, 
                "fea": fea_one, "exp": exp_one}


class SourceDataSetLoop(data.Dataset):
    def __init__(self, seqs, motifs, features, expression, loop):
        super(SourceDataSetLoop, self).__init__()
        self.seq = seqs
        self.motif = motifs
        self.fea = features
        self.exp = expression
        self.loop = loop

        assert len(self.seq) == len(self.exp) and len(self.fea) == len(self.exp), \
        "the number of sequences and labels must be consistent."

    def __len__(self):
        return len(self.exp)

    def __getitem__(self, index):
        seq_one = self.seq[index]
        motif_one = self.motif[index]
        fea_one = self.fea[index]
        exp_one = self.exp[index]
        loop_one = self.loop[index]

        return {"seq": seq_one, "motif": motif_one, "fea": fea_one, 
                "exp": exp_one, "loop": loop_one}