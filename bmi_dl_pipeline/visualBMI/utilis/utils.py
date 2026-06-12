
import torch
import yaml
import json
import numpy as np
import random
import os


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def save_model(model, path, config=None):
    torch.save(model.state_dict(), path)
    if config:
        cfg_path = path.replace('.pth', '_config.yaml')
        with open(cfg_path, 'w') as f:
            yaml.dump(config, f)
    print(f"✅ Model saved: {path}")


def load_model(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model


def save_results(results_list, path):
    import pandas as pd
    pd.DataFrame(results_list).to_csv(path, index=False)
    print(f"✅ Results saved: {path}")


def save_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)
    print(f"✅ JSON saved: {path}")
