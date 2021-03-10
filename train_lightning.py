import pytorch_lightning as pl
from torch.utils.tensorboard import SummaryWriter

from models.retinanet import RetinanetLightning

writer = SummaryWriter()
model = RetinanetLightning(classes_cnt=21, tboard_writer=writer)

trainer = pl.Trainer(limit_val_batches=1,
                     #gpus=[0],
                     val_check_interval=20,
                     check_val_every_n_epoch=1)
trainer.fit(model)
