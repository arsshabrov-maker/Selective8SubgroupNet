import operator
import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits, load_wine, make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import Selective8SubgroupNet


class BaselineMLP(nn.Module):

  def __init__(self, in_dim, out_dim, width=33, depth=4, use_activation=True):
    super().__init__()
    self.use_activation = use_activation
    self.in_proj = nn.Linear(in_dim, width)
    self.layers = nn.ModuleList(
        [nn.Linear(width, width) for _ in range(depth - 1)]
    )
    self.head = nn.Linear(width, out_dim)

  def forward(self, x):
    h = F.gelu(self.in_proj(x)) if self.use_activation else self.in_proj(x)
    for layer in self.layers:
      z = layer(h)
      h = F.gelu(z) if self.use_activation else z
    return self.head(h)


def build_4_tasks():
  tasks = []
  X1, y1 = make_circles(n_samples=900, noise=0.10, factor=0.45, random_state=42)
  tasks.append(('1. Nested Circles (Nonlinear)', X1, y1))
  d2 = load_digits()
  tasks.append(('2. Digits 8x8 (Vision)', d2.data, d2.target))
  d3 = load_breast_cancer()
  tasks.append(('3. Breast Cancer (Medical)', d3.data, d3.target))
  d4 = load_wine()
  tasks.append(('4. Wine Chem (Tabular)', d4.data, d4.target))

  out = []
  for name, X_r, y_r in tasks:
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_r, y_r, test_size=0.25, stratify=y_r, random_state=42
    )
    sc = StandardScaler()
    out.append((
        name,
        torch.tensor(sc.fit_transform(X_tr), dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.long),
        torch.tensor(sc.transform(X_te), dtype=torch.float32),
        torch.tensor(y_te, dtype=torch.long),
        len(np.unique(y_r)),
    ))
  return out


def main():
  tasks = build_4_tasks()
  variants = [
      ('1. Linear MLP (No Activation)', lambda i, o: BaselineMLP(i, o, 33, 4, False)),
      ('2. 8-Subgroup Net (NO Activation)', lambda i, o: Selective8SubgroupNet(i, o, 32, 4, False)),
      ('3. Standard MLP (+ GELU)', lambda i, o: BaselineMLP(i, o, 33, 4, True)),
      ('4. 8-Subgroup Net (+ GELU)', lambda i, o: Selective8SubgroupNet(i, o, 32, 4, True)),
  ]

  summary = {t: [] for t, _ in variants}
  for name, X_tr, y_tr, X_te, y_te, n_o in tasks:
    print(f'\n--- {name} ---')
    for title, builder in variants:
      accs = []
      for sd in (42, 100, 777):
        torch.manual_seed(sd)
        net = builder(X_tr.size(1), n_o)
        n_p = sum(p.numel() for p in net.parameters())
        opt = torch.optim.AdamW(net.parameters(), lr=0.008, weight_decay=1e-3)
        best = 0.0
        for _ in range(160):
          net.train()
          opt.zero_grad()
          F.cross_entropy(net(X_tr), y_tr).backward()
          opt.step()
          net.eval()
          with torch.no_grad():
            best = max(
                best,
                (net(X_te).argmax(dim=1) == y_te).float().mean().item() * 100.0,
            )
        accs.append(best)
      m_a = float(np.mean(accs))
      summary[title].append(m_a)
      print(f'{title:<38} | Params: {n_p:<5} | Mean Acc (3 seeds): {m_a:5.2f}%')


if __name__ == '__main__':
  main()
