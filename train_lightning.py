import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from models.retinanet import RetinanetLightning


checkpoint_callback = ModelCheckpoint(
    save_top_k=3,
    verbose=True,
    monitor="val_loss",
    mode="min"
)
logs_root = "logs"
logger = TensorBoardLogger(logs_root, name='my_model')
print(f"Writer initialized at {logs_root}")

model = RetinanetLightning(classes_cnt=21,
                           train_batch=32,
                           val_batch=64,
                           dataset_storage=None)
print(f"Model has been built {model}")


trainer = pl.Trainer(max_epochs=5,
                     limit_val_batches=1,
                     #gpus=1,
                     val_check_interval=0.25,
                     callbacks=[checkpoint_callback])

trainer.fit(model)
