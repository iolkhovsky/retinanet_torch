import pytorch_lightning as pl
from torch.utils.tensorboard import SummaryWriter

from models.ssd_mobilenet_v2 import SSDLightning

writer = SummaryWriter()
model = SSDLightning(classes_cnt=21, tboard_writer=writer)

trainer = pl.Trainer(limit_val_batches=1,
                     #gpus=[0],
                     val_check_interval=50,
                     check_val_every_n_epoch=1)
trainer.fit(model)
