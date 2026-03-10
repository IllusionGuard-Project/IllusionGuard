# train_diagnoser.py
import os, numpy as np, torch, torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader

CLASSES = None  # 训练时自动从数据集推断并保存

class NpzSet(Dataset):
    def __init__(self, npz_path, le=None):
        d = np.load(npz_path, allow_pickle=True)
        self.X = d["X"].astype("float32")
        self.y_str = d["y"].astype(str)
        if le is None:
            self.le = LabelEncoder().fit(self.y_str)
        else:
            self.le = le
        self.y = self.le.transform(self.y_str).astype("int64")
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

class MLP(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, 64), nn.GELU(),
            nn.Linear(64, 32), nn.GELU(),
            nn.Linear(32, d_out)
        )
    def forward(self, x): return self.net(x)

def main(train_npz, val_npz, out_pt="ckpts/diag_cls.pt", epochs=15, bs=128, lr=2e-3, device="cuda"):
    os.makedirs(os.path.dirname(out_pt), exist_ok=True)

    # 载入数据（推断标签空间）
    ds_tr = NpzSet(train_npz)
    le = ds_tr.le
    ds_va = NpzSet(val_npz, le=le)

    dl_tr = DataLoader(ds_tr, batch_size=bs, shuffle=True)
    dl_va = DataLoader(ds_va, batch_size=bs, shuffle=False)

    model = MLP(d_in=ds_tr.X.shape[1], d_out=len(le.classes_)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    ce  = nn.CrossEntropyLoss()

    best = (0.0, None)
    for ep in range(1, epochs+1):
        model.train(); tot=0
        for x,y in dl_tr:
            x,y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = ce(model(x), y)
            loss.backward(); opt.step()
            tot += loss.item()*x.size(0)
        tr_loss = tot/len(ds_tr)

        # val
        model.eval(); correct=0
        with torch.no_grad():
            for x,y in dl_va:
                x,y = x.to(device), y.to(device)
                pred = model(x).argmax(-1)
                correct += (pred==y).sum().item()
        acc = correct/len(ds_va)
        print(f"[Ep{ep}] train_loss={tr_loss:.4f}  val_acc={acc:.4f}")

        if acc > best[0]:
            best = (acc, model.state_dict())
            torch.save({
                "state": best[1],
                "classes": le.classes_.tolist(),
                "din": ds_tr.X.shape[1],
            }, out_pt)
            print(f"  -> saved {out_pt} (best_acc={acc:.4f})")

if __name__ == "__main__":
    import argparse; ap=argparse.ArgumentParser()
    ap.add_argument("--train_npz", required=True)
    ap.add_argument("--val_npz",   required=True)
    ap.add_argument("--out", default="ckpts/diag_cls.pt")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    main(args.train_npz, args.val_npz, args.out, args.epochs, args.bs, args.lr, args.device)
