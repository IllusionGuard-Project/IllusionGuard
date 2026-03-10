# predict_diagnoser.py
import os, csv, numpy as np, torch, torch.nn as nn

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

def main(npz_path, ckpt, out_csv):
    d = np.load(npz_path, allow_pickle=True)
    X = d["X"].astype("float32")
    paths = d["paths"].astype(str)

    ck = torch.load(ckpt, map_location="cpu")
    din, classes = ck["din"], ck["classes"]
    model = MLP(d_in=din, d_out=len(classes))
    model.load_state_dict(ck["state"]); model.eval()

    with torch.no_grad(), open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["image","pred_label","probs_json"])
        logits = model(torch.tensor(X))
        prob = torch.softmax(logits, dim=-1).numpy()
        for i, p in enumerate(prob):
            idx = int(np.argmax(p))
            w.writerow([paths[i], classes[idx], {classes[j]: float(p[j]) for j in range(len(classes))}])

    print(f"✅ 写入预测结果: {out_csv}")

if __name__ == "__main__":
    import argparse; ap=argparse.ArgumentParser()
    ap.add_argument("--npz",  required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out",  required=True)
    args = ap.parse_args()
    main(args.npz, args.ckpt, args.out)
