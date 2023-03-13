from torch import Tensor, nn


class ContextEmbedding(nn.Module):
    def __init__(self, emb_dim: int, protected_attr: str):
        super().__init__()
        assert protected_attr in ['sex', 'age']
        self.protected_attr = protected_attr
        if self.protected_attr == 'sex':
            self.emb = nn.Embedding(2, emb_dim)
        else:
            self.emb = nn.Linear(1, emb_dim)

    def forward(self, context: Tensor) -> Tensor:
        if self.protected_attr == 'age':
            context = (context - 50) / 100
            context = context.unsqueeze(1)
        return self.emb(context)
