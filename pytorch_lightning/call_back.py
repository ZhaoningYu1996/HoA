from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.callbacks import Callback

class MyCallback(Callback):
    def __init__(self, monitor='val_rocauc_score', patience=20, verbose=True, mode='max', checkpoint_dir='checkpoints/'):
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.mode = mode
        
        # Checkpoint
        self.checkpoint_callback = ModelCheckpoint(
            monitor=monitor,
            dirpath=checkpoint_dir,
            filename='best_model',
            save_top_k=1,
            mode=mode
        )
        
        # Early Stopping
        self.early_stop_callback = EarlyStopping(
            monitor=monitor,
            min_delta=0.00,
            patience=patience,
            verbose=verbose,
            mode=mode
        )

        # Progress Bar
        # self.progress_bar = EpochProgressBar()

        self.call_backs = [self.checkpoint_callback]

