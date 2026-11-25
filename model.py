import torch
import torch.nn as nn
import torch.nn.functional as F


class Selective8SubgroupNet(nn.Module):
  """
    Selective 8-Subgroup Neural Architecture (`Selective8SubgroupNet`):
      Demonstrates that structural routing alone—without any pointwise activation
      function (GELU/ReLU)—generates full universal nonlinearity.

      Each layer partitions units into 8 specialized subgroups (2 Polarities x 4 Stopping Rules):
        - Group A (Subgroups 0..3): Large activations -> Width; Small -> Depth
        - Group B (Subgroups 4..7): Small activations -> Width; Large -> Depth
        - Stopping Rules when propagating deep to target neuron h_dst:
            0 ('≈'): Resonance stop when |h_src - h_dst| ~ 0
            1 ('<'): Lower-bound stop when h_src < h_dst
            2 ('>'): Upper-bound stop when h_src > h_dst
            3 ('∞'): Unconditional pass-through
    """

  def __init__(self, in_dim, out_dim, width=32, depth=4, use_activation=False):
    super().__init__()
    assert width % 8 == 0, 'Layer width must be divisible by 8 subgroups.'
    self.width = width
    self.depth = depth
    self.use_activation = use_activation

    self.in_proj = nn.Linear(in_dim, width)
    self.width_layers = nn.ModuleList(
        [nn.Linear(width, width) for _ in range(depth - 1)]
    )
    self.depth_weights = nn.ParameterList([
        nn.Parameter(torch.randn(depth - 1 - l, width) * 0.35 + 0.5)
        for l in range(depth - 1)
    ])
    self.balance_thresh = nn.ParameterList(
        [nn.Parameter(torch.ones(1, width) * 0.5) for _ in range(depth - 1)]
    )
    self.stop_sharpness = nn.ParameterList(
        [nn.Parameter(torch.ones(1, width) * 1.5) for _ in range(depth - 1)]
    )

    sub_size = width // 8
    pol_list = []
    rule_list = []
    for g in range(8):
      pol = 1.0 if g < 4 else -1.0
      rule = g % 4
      pol_list.extend([pol] * sub_size)
      rule_list.extend([rule] * sub_size)

    self.register_buffer(
        'pol_mask', torch.tensor(pol_list, dtype=torch.float32).view(1, width)
    )
    self.register_buffer(
        'rule_mask', torch.tensor(rule_list, dtype=torch.long).view(1, width)
    )
    self.head = nn.Linear(width, out_dim)

  def forward(self, x):
    B = x.size(0)
    layer_buf = [self.in_proj(x)] + [
        torch.zeros(B, self.width, device=x.device)
        for _ in range(self.depth - 1)
    ]

    for l in range(self.depth - 1):
      h = F.gelu(layer_buf[l]) if self.use_activation else layer_buf[l]

      # 1. Edge-budget conservation between Width and Depth
      mag = torch.abs(h)
      thresh = F.softplus(self.balance_thresh[l])
      alpha_a = torch.sigmoid(2.5 * (mag - thresh))
      alpha_b = 1.0 - alpha_a

      w_mix = 0.5 * (self.pol_mask + 1.0)
      alpha = w_mix * alpha_a + (1.0 - w_mix) * alpha_b

      w_gate = torch.sqrt(alpha + 1e-6)
      d_gate = torch.sqrt(1.0 - alpha + 1e-6)

      # Lateral stream to next layer
      layer_buf[l + 1] = layer_buf[l + 1] + self.width_layers[l](h * w_gate)

      # 2. Selective depth propagation with 4 value-comparison stopping rules
      long_sig = h * d_gate
      sharp = F.softplus(self.stop_sharpness[l]) + 0.5
      d_weights = self.depth_weights[l]

      for step in range(self.depth - 1 - l):
        target_l = l + 1 + step
        h_dst = layer_buf[target_l].detach()
        diff = h - h_dst

        gate_eq = torch.exp(-sharp * (diff**2))
        gate_lt = torch.sigmoid(sharp * (-diff))
        gate_gt = torch.sigmoid(sharp * diff)
        gate_all = torch.ones_like(h)

        r = self.rule_mask
        stop_gate = torch.where(
            r == 0,
            gate_eq,
            torch.where(r == 1, gate_lt, torch.where(r == 2, gate_gt, gate_all)),
        )

        delivered = long_sig * stop_gate * d_weights[step].unsqueeze(0)
        layer_buf[target_l] = layer_buf[target_l] + delivered
        long_sig = long_sig * (1.0 - 0.5 * stop_gate)

    out_h = F.gelu(layer_buf[-1]) if self.use_activation else layer_buf[-1]
    return self.head(out_h)
