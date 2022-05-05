download_data:
	python PNGAN/util/download_data.py --data train-test --dataset SIDD --noise real

preprocessed:
	python PNGAN/util/generate_patches_sidd.py
	python PNGAN/util/generate_patches_sidd_val.py

train_pngan:
	cd PNGAN && python main.py

train_with_best:
	cd PNGAN && python main.py --load_models --load_best --load_dir $(load_dir)

train_with_epoch:
	cd PNGAN && python main.py --load_models --load_dir $(load_dir) --load_epoch $(epoch_num)

test_pngan_with_best:
	cd PNGAN && python main.py --test_only --load_models --load_best --load_dir $(load_dir)

test_pngan_with_epoch:
	cd PNGAN && python main.py --load_models --load_dir $(load_dir) --load_epoch $(epoch_num) --test_only



