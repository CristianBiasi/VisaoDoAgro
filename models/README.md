# Custom model weights

Place a future trained model at `models/best.pt`, then set `model_path` to
`models/best.pt` in Advanced Settings. No custom weights are included.

The existing generic `yolo11n.pt` remains at the repository root for compatibility.
Class names come from the model metadata. An empty monitored class list selects all
classes; unknown configured names are reported and excluded.
