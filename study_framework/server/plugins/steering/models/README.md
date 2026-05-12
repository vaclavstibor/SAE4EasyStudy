# Steering Models

This directory is owned by the `server/plugins/steering/` plugin.

Required local model artifact for the current SAE study:

- `TopKSAE-1024.ckpt` or `TopKSAE-1024.pt`

Large binaries stay ignored by git. Populate this directory manually or by the
supported model bootstrap command:

```bash
python -m server.plugins.steering.bootstrap_model
```
