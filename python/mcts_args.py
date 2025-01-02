from dataclasses import dataclass


@dataclass
class MctsArgs:
    # These values are decided from hyperparameter sweeping not very thorough
    update_threshold: float = 0.53
    do_arena: bool = True
    # Revert the model if rejected
    enable_rejecting: bool = False
    # if False, compare with a basically random player in the arena
    compare_with_self: bool = False
    # In the original implementation, randomly choose the training data
    random_choice: bool = False
    
    # Newer is better
    num_iters_for_train_examples_history: int = 1

    checkpoint: str = "./temp/"
    load_model: bool = False
    load_folder_file: tuple[str, str] = (
        "/dev/models/8x100x50", "best.pth.tar")

    is_release: bool = False

    lr: float = 1e-4

    dropout: float = 0.1
    # The lower the better
    epochs: int = 1
    cuda: bool = True
    num_channels: int = 512
