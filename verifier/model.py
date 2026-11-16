"""Test-time verifier: one language-conditioned model for all tasks (prediction horizon 10, 5,028,969 parameters).

State s_t = [phi_obs(o_t) (agent + wrist DINOv2 features, 768) ; phi_lang(l) (960 -> 64 learned projection) ; p_t (8) ; j_t (18)].
Dynamics MLP f_dyn(s_t, a): input layer, 4 residual MLP blocks (width 512, LayerNorm, ReLU, dropout), per-step change of the
robot state (H x 26), accumulated on top of the current robot state. State-attention classifier f_cls: learned positions,
one self-attention layer (8 heads, width 256), FFN 512, learned-query pooling, head 256 -> 128 -> 64 -> 1.
Inputs and predicted changes are standardised with fixed (non-trained) statistics stored as buffers.
"""
from dataclasses import dataclass, asdict
import math
import torch
from torch import nn
from torch.nn import functional as F

IMPLEMENTATION = 'verifier-h10-v1'


@dataclass(frozen=True)
class VerifierConfig:
    horizon: int = 10
    proprio_dim: int = 8
    joint_dim: int = 18
    action_dim: int = 7
    image_dim: int = 768
    language_dim: int = 960
    language_projection: int = 64
    hidden: int = 512
    dynamics_inner: int = 808
    blocks: int = 4
    model_dim: int = 256
    heads: int = 8
    ffn_dim: int = 512
    head1: int = 128
    head2: int = 64
    dropout: float = .1

    @property
    def robot_dim(self): return self.proprio_dim + self.joint_dim
    @property
    def state_dim(self): return self.robot_dim + self.image_dim + self.language_projection


class DynamicsMLP(nn.Module):
    def __init__(self, c):
        super().__init__(); self.c = c
        self.input = nn.Linear(c.state_dim + c.horizon * c.action_dim, c.hidden)
        self.blocks = nn.ModuleList([nn.Sequential(nn.LayerNorm(c.hidden), nn.Linear(c.hidden, c.dynamics_inner), nn.ReLU(),
                                                   nn.Linear(c.dynamics_inner, c.hidden), nn.Dropout(c.dropout)) for _ in range(c.blocks)])
        self.output = nn.Linear(c.hidden, c.horizon * c.robot_dim)

    def forward(self, state, actions):
        h = self.input(torch.cat([state, actions.flatten(1)], -1))
        for block in self.blocks: h = h + block(h)
        return self.output(h).view(-1, self.c.horizon, self.c.robot_dim)


class StateAttentionClassifier(nn.Module):
    def __init__(self, c):
        super().__init__(); self.c = c
        self.position = nn.Parameter(torch.randn(1, c.horizon, c.state_dim) * .02)
        self.q = nn.Linear(c.state_dim, c.model_dim); self.k = nn.Linear(c.state_dim, c.model_dim); self.v = nn.Linear(c.state_dim, c.model_dim)
        self.output = nn.Linear(c.model_dim, c.model_dim)
        self.norm1 = nn.LayerNorm(c.model_dim); self.norm2 = nn.LayerNorm(c.model_dim)
        self.dropout = nn.Dropout(c.dropout)
        self.ffn = nn.Sequential(nn.Linear(c.model_dim, c.ffn_dim), nn.ReLU(), nn.Dropout(c.dropout), nn.Linear(c.ffn_dim, c.model_dim))
        self.pool_query = nn.Parameter(torch.randn(c.model_dim) * .02)
        self.head = nn.Sequential(nn.Linear(c.model_dim, c.head1), nn.ReLU(), nn.Dropout(c.dropout),
                                  nn.Linear(c.head1, c.head2), nn.ReLU(), nn.Linear(c.head2, 1))

    def forward(self, sequence):
        x = sequence + self.position
        q, k, v = self.q(x), self.k(x), self.v(x)
        def split(t): return t.view(t.shape[0], t.shape[1], self.c.heads, -1).transpose(1, 2)
        h = F.scaled_dot_product_attention(split(q), split(k), split(v), dropout_p=self.c.dropout if self.training else 0.)
        h = h.transpose(1, 2).reshape(q.shape)
        h = self.norm1(q + self.dropout(self.output(h)))
        h = self.norm2(h + self.dropout(self.ffn(h)))
        w = (h @ self.pool_query / math.sqrt(self.c.model_dim)).softmax(1)
        return self.head((h * w[:, :, None]).sum(1)).squeeze(-1)


class Verifier(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or VerifierConfig(); c = self.config
        self.language = nn.Linear(c.language_dim, c.language_projection)
        self.dynamics = DynamicsMLP(c)
        self.classifier = StateAttentionClassifier(c)
        for name, dim in [('robot', c.robot_dim), ('image', c.image_dim), ('action', c.action_dim), ('lang', c.language_dim), ('delta', c.robot_dim)]:
            self.register_buffer(f'{name}_mean', torch.zeros(dim)); self.register_buffer(f'{name}_std', torch.ones(dim))

    def set_stats(self, **stats):
        with torch.no_grad():
            for name, (mean, std) in stats.items():
                getattr(self, f'{name}_mean').copy_(torch.as_tensor(mean)); getattr(self, f'{name}_std').copy_(torch.as_tensor(std).clamp_min(1e-4))

    def forward(self, robot, image, language, actions):
        c = self.config
        if actions.shape[1:] != (c.horizon, c.action_dim):
            raise ValueError(f'Expected actions (B,{c.horizon},{c.action_dim}), got {tuple(actions.shape)}')
        r = (robot - self.robot_mean) / self.robot_std
        image_n = (image - self.image_mean) / self.image_std; lang_n = self.language((language - self.lang_mean) / self.lang_std)
        context = torch.cat([image_n, lang_n], -1)
        delta_n = self.dynamics(torch.cat([context, r], -1), (actions - self.action_mean) / self.action_std)
        delta = delta_n * self.delta_std + self.delta_mean
        predicted = robot[:, None, :] + delta.cumsum(1)
        sequence = torch.cat([(predicted - self.robot_mean) / self.robot_std, context[:, None, :].expand(-1, c.horizon, -1)], -1)
        return self.classifier(sequence), delta, predicted

    def interface(self):
        c = self.config
        return dict(horizon=c.horizon, proprio_dim=c.proprio_dim, joint_dim=c.joint_dim, action_dim=c.action_dim, image_dim=c.image_dim, language_dim=c.language_dim)

    def parameter_report(self):
        return {'implementation': IMPLEMENTATION, 'total': sum(p.numel() for p in self.parameters()),
                'modules': {n: sum(p.numel() for p in m.parameters()) for n, m in self.named_children()}, 'config': asdict(self.config)}
