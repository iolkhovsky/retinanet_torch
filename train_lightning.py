import pytorch_lightning as pl

from models.ssd_mobilenet_v2 import SSDLightning


model = SSDLightning(classes_cnt=21)

trainer = pl.Trainer()
trainer.fit(model)
