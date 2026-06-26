# Curated Checkpoints

This folder stores only the checkpoints required to reproduce final tables and claims.

- `cold10_llminit_var_no_connector.pth`: Cold-10 main table baseline (authors/no-connector LLMInit-Var on `amazon-beauty-cold10core`).
- `cold10_plain_connector_var.pth`: Cold-10 main table plain connector result on `amazon-beauty-cold10core`.
- `cold10_residual_connector_var_alpha01.pth`: Cold-10 main table residual connector result (`alpha=0.1`) on `amazon-beauty-cold10core`.
- `low5_plain_connector_var_norefilter.pth`: Low-5 appendix table plain connector (no-residual) run2 on `amazon-beauty-low5cold`.
- `low5_residual_connector_var_alpha01_norefilter.pth`: Low-5 appendix table residual connector (`alpha=0.1`) run2 on `amazon-beauty-low5cold`.
- `beauty_plain_connector_var.pth`: Fairness connector row on standard `amazon-beauty` with plain connector.
- `beauty_residual_connector_var_alpha01.pth`: Fairness connector row on standard `amazon-beauty` with residual connector (`alpha=0.1`).

Optional ablation checkpoint is intentionally excluded unless explicitly reported in the paper/report: `saved/ContGCN-Jun-24-2026_23-21-11.pth`.
