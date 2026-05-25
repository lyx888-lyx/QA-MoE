import os, torch

# def save_checkpoint(model, optimizer, epoch, save_dir, suffix=""):
#     os.makedirs(save_dir, exist_ok=True)
#     path = os.path.join(save_dir, f"checkpoint_{suffix}_epoch{epoch}.pt")
#     torch.save({
#         "model": model.state_dict(),
#         "optimizer": optimizer.state_dict(),
#         "epoch": epoch
#     }, path)

# def load_checkpoint(model, optimizer, path, device):
#     checkpoint = torch.load(path, map_location=device)
#     model.load_state_dict(checkpoint["model"])
#     optimizer.load_state_dict(checkpoint["optimizer"])
#     return checkpoint["epoch"] + 1

def save_checkpoint(model, optimizer, epoch, path):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch
    }, path)


def load_checkpoint(model, optimizer, path, device='cuda'):
    if not os.path.exists(path):
        print(f"Checkpoint not found: {path}")
        return 0

    state = torch.load(path, map_location=device)
    model.load_state_dict(state['model_state'])
    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer_state'])
    print(f"Loaded checkpoint from {path}, epoch {state['epoch']}")
    return state['epoch']
