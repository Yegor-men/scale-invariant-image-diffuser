import argparse, json, tensorboard
from tqdm import tqdm


def create_scheduler():
    pass


def update_ema_model(model, ema_model, decay):
    pass


def create_dataloader(batch_size):
    pass


def save_model(ema_model, path):
    pass


def train(num_epochs, batch_accum_size, train_dataloader, test_dataloader, model, ema_model, device):
    for e in range(num_epochs):
        for image, label in tqdm(train_dataloader, total=len(train_dataloader), desc=f"E{e + 1} - TRAIN"):
            image, label = image.to(device), label.to(device)



        for image, label in enumerate(test_dataloader):
            pass

    save_model(model, ema_model)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="models/example/config.json")
    p.add_argument("--load_existing", type=bool, default=False)  # Detect if a model already exists there

    p.add_argument("--base_image_size", type=int, default=64)
    p.add_argument("--random_image_resize", type=bool, default=False)
    p.add_argument("--min_random_size", type=int, default=48)
    p.add_argument("--max_random_size", type=int, default=80)

    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--num_epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=50)
    p.add_argument("--batch_accum_size", type=int, default=1)
    p.add_argument("--max_lr", type=float, default=1e-3)
    p.add_argument("--min_lr", type=float, default=1e-5)
    p.add_argument("--warmup_steps", type=int, default=1200)

    p.add_argument("--tensorboard_port", type=int, default=6006)

    p.add_argument("--device", type=str, default="auto")


def main():
    args = get_args()
    model = SIID()
    train()
