import torch
import lightning as L
from data_module import OGBDataModule
from pl_module import Encoder
from gnn_module import GNNEncoder
from call_back import MyCallback
import numpy as np
from lightning.pytorch.loggers import WandbLogger
import wandb

# Set float32 matmul precision to 'high' to utilize Tensor Cores
torch.set_float32_matmul_precision('high')

wandb.login(key="cef0853f1b3259a908b6907a40981bd4fd274708")

data_name = "ogbg-molbbbp"
data_module = OGBDataModule(data_name=data_name, data_dir="data/", batch_size=32)
data_module.setup()

num_runs = 10
test_rocauc = []

for i in range(num_runs):
    # model = Encoder(data_name=data_name, input_dim=data_module.feat_dim, hidden_dim=300, output_dim=data_module.out_dim)
    model = GNNEncoder(data_name=data_name, num_layers=5, hidden_dim=300, output_dim=data_module.out_dim, dropout=0.5)
    my_callback = MyCallback(monitor='val_rocauc_score', patience=20, verbose=True, mode='max', checkpoint_dir='checkpoints/')

    wandb_logger = WandbLogger(name=f'hoa_bbbp_run_{i+1}', project='HoA')

    # train model
    trainer = L.Trainer(max_epochs=100, callbacks=my_callback.call_backs, logger=wandb_logger, enable_progress_bar=False)
    trainer.fit(model=model, datamodule=data_module)
    trainer.test(model=model, datamodule=data_module)
    test_rocauc.append(trainer.logged_metrics['test_rocauc_score'].item())
    wandb_logger.experiment.finish()

test_rocauc = np.array(test_rocauc)
mean_rocauc = np.mean(test_rocauc)
std_rocauc = np.std(test_rocauc)

print(f"Mean Test Accuracy: {mean_rocauc}")
print(f"Standard Deviation of Test Accuracy: {std_rocauc}")