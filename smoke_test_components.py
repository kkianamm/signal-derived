import torch
from models.semantic_qformer_components import (
    AttentionPool, GatedQueryResidual, SemanticQueryFormer,
    query_diversity_loss, supervised_contrastive_alignment,
)

B, NMED, NIMG, D, Q, K = 4, 31, 49, 512, 32, 5
med = torch.randn(B, NMED, D)
img = torch.randn(B, NIMG, D)
memory = torch.cat([med, img], dim=1)
semantic = torch.randn(K, D)
qf = SemanticQueryFormer(D, total_queries=Q, semantic_queries=K, depth=2, heads=8)
queries, attn = qf(memory, semantic)
assert queries.shape == (B, Q, D)
assert len(attn) == 2
pool = AttentionPool(D)
med_repr = pool(med)
img_repr = pool(img)
gate = GatedQueryResidual(D)
gated, weights = gate(queries, med_repr)
assert gated.shape == (B, Q, D)
assert weights.shape == (B, Q, D)
labels = torch.tensor([0, 0, 1, 2])
loss = supervised_contrastive_alignment(med_repr, img_repr, labels) + query_diversity_loss(gated)
assert torch.isfinite(loss)
print("OK", {"queries": tuple(queries.shape), "gate": tuple(weights.shape), "loss": float(loss)})
