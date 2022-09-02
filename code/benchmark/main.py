import torch
import tsl
import numpy as np
from tsl.datasets import MetrLA
from tsl.data import SpatioTemporalDataset
from tsl.data import SpatioTemporalDataModule
from tsl.data.preprocessing import StandardScaler
from tsl.nn.metrics.metrics import MaskedMAE, MaskedMAPE
from tsl.predictors import Predictor
from model import TimeThenSpaceModel
from pytorch_lightning.loggers import CSVLogger
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import seaborn
from utility.correlation import Correlation
import random
from tsl.ops.connectivity import adj_to_edge_index
import os

seaborn.set()

if __name__ == '__main__':

    # reproducibility
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    os.environ['PYTHONHASHSEED'] = str(0)
    # log suppression
    np.set_printoptions(suppress=True)
    tsl.logger.disabled = True

    # dataset creation
    dataset = MetrLA()

    correlation = Correlation(dataset)

    for method in correlation.get_correlation_methods():
        for n_exp in range(5):
            if os.path.isfile('numpy data/' + method + '.npy'):
                adj = np.load('numpy data/' + method + '.npy')
            else:
                adj = correlation.get_correlation(
                    method=method,
                    threshold=0.1,
                    include_self=False,
                    normalize_axis=1,
                    layout="dense"
                )
                np.save('numpy data/' + method, adj)

            adj = adj_to_edge_index(adj)

            torch_dataset = SpatioTemporalDataset(*dataset.numpy(return_idx=True),
                                                  connectivity=adj,
                                                  mask=dataset.mask,
                                                  horizon=12,
                                                  window=12)

            scalers = {'data': StandardScaler(axis=(0, 1))}

            splitter = dataset.get_splitter(val_len=0.1, test_len=0.2)

            dm = SpatioTemporalDataModule(
                dataset=torch_dataset,
                scalers=scalers,
                splitter=splitter,
                batch_size=64
            )

            dm.setup()

            # Train

            loss_fn = MaskedMAE(compute_on_step=True)

            metrics = {
                'mae': MaskedMAE(compute_on_step=False),
                'mape': MaskedMAPE(compute_on_step=False),
                'mae_at_15': MaskedMAE(compute_on_step=False, at=2),
                'mae_at_30': MaskedMAE(compute_on_step=False, at=5),
                'mae_at_60': MaskedMAE(compute_on_step=False, at=11),
            }

            model_kwargs = {
                'input_size': dm.n_channels,  # 1 channel
                'horizon': dm.horizon,  # 12, the number of steps ahead to forecast
                'hidden_size': 32,
                'rnn_layers': 1,
                'gcn_layers': 2
            }

            # setup predictor
            predictor = Predictor(
                model_class=TimeThenSpaceModel,
                model_kwargs=model_kwargs,
                optim_class=torch.optim.Adam,
                optim_kwargs={'lr': 0.001},
                loss_fn=loss_fn,
                metrics=metrics
            )

            logger = CSVLogger(save_dir="logs", name=method)

            checkpoint_callback = ModelCheckpoint(
                dirpath='logs',
                save_top_k=1,
                monitor='val_mae',
                mode='min',
            )

            trainer = pl.Trainer(max_epochs=2,
                                 logger=logger,
                                 gpus=1 if torch.cuda.is_available() else None,
                                 limit_train_batches=100,
                                 callbacks=[
                                     checkpoint_callback,
                                     EarlyStopping(monitor="val_loss",
                                                   mode="min", patience=3)])

            trainer.fit(predictor, datamodule=dm)

            predictor.load_model(checkpoint_callback.best_model_path)
            predictor.freeze()

            performance = trainer.test(predictor, datamodule=dm)
        print(performance)
