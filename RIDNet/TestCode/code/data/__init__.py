from importlib import import_module
from torch.utils.data.dataloader import default_collate
from torch.utils.data import DataLoader
from dataset import *


class Data:
    def __init__(self, args):
        train_ds = SIDDSmallDataset(args.dir_data, noise_generator=AdditiveGaussianWhiteNoise(std=50.),
                                    random_load=True, limit=args.n_train)
        val_ds = SIDDSmallDataset(args.dir_data, data_type='val',
                                  noise_generator=AdditiveGaussianWhiteNoise(std=50.),
                                  random_load=True, limit=args.n_val)
        self.loader_train = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
        self.loader_test = DataLoader(val_ds, batch_size=args.batch_size, num_workers=4)
        return
        kwargs = {}
        if not args.cpu:
            kwargs['collate_fn'] = default_collate
            kwargs['pin_memory'] = True
        else:
            kwargs['collate_fn'] = default_collate
            kwargs['pin_memory'] = False

        self.loader_train = None
        if not args.test_only:
            module_train = import_module('dataset.' + args.data_train.lower())
            trainset = getattr(module_train, args.data_train)(args)
            self.loader_train = DataLoader(
                trainset,
                batch_size=args.batch_size,
                shuffle=True
            )

        if args.data_test in ['CBSD68', 'Set14', 'B100', 'Urban100']:
            if not args.benchmark_noise:
                module_test = import_module('dataset.benchmark')
                testset = getattr(module_test, 'Benchmark')(args, train=False)
            else:
                module_test = import_module('dataset.benchmark_noise')
                testset = getattr(module_test, 'BenchmarkNoise')(
                    args,
                    train=False
                )

        else:
            module_test = import_module('dataset.' + args.data_test.lower())
            testset = getattr(module_test, args.data_test)(args, train=False)

        self.loader_test = DataLoader(
            testset,
            batch_size=1,
            shuffle=False,
        )
