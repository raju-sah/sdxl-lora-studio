| Checkpoint | CLIP Content Score ↑ | DINOv2 Style Sim ↑ | DINOv2 Max-NN (Memorization) | Status |
|---|---:|---:|---:|---|
| Base SDXL (0) | 0.2961 | 0.1865 | 0.3956 | Baseline |
| `ckpt-250` | 0.2718 | 0.3728 | 0.5945 | Underfit / warming up |
| `ckpt-500` | 0.2501 | 0.3544 | 0.5937 | Good |
| `ckpt-750` | 0.2666 | 0.3949 | 0.6317 | ⭐ **Selected Best** |
| `ckpt-1000` | 0.2588 | 0.3694 | 0.6043 | Good |
| `ckpt-1250` | 0.2568 | 0.3957 | 0.6263 | Good |
| `ckpt-1500` | 0.2641 | 0.3814 | 0.6214 | Late-stage saturation |
