
from tensorboard.backend.event_processing import event_accumulator
 
#加载日志数据
ea=event_accumulator.EventAccumulator('runs/Mar17_04-10-48_06bed19cdc6a/loss_BCE/events.out.tfevents.1615954259.06bed19cdc6a.8881.2') 
ea.Reload()

val_psnr=ea.scalars.Items('loss')


BCE = [i.value for i in val_psnr]
