import torch
import utility
import data
import model
import loss
from option import args
from trainer import Trainer
from torch.nn import DataParallel

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

if checkpoint.ok:
    loader = data.Data(args)
    model = model.Model(args, checkpoint)
    model_data = DataParallel(model)
    loss = loss.Loss(args, checkpoint) if not args.test_only else None
    t = Trainer(args, loader, model_data, loss, checkpoint)
    while not t.terminate():
        if args.test_only:
            t.test()
        else:
            t.train()

    checkpoint.done()
