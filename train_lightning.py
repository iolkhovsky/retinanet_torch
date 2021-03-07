import pytorch_lightning as pl

from models.ssd_mobilenet_v2 import SSDLightning


model = SSDLightning(classes_cnt=21)

trainer = pl.Trainer(limit_val_batches=1,
                     val_check_interval=50,
                     check_val_every_n_epoch=1)
trainer.fit(model)
