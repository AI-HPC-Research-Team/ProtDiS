import argparse, os, sys, datetime, shutil
import random
import copy
import time
import importlib

import logging
import matplotlib.pyplot as plt

import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch_geometric.loader import DataLoader
from omegaconf import OmegaConf
from tqdm import tqdm
import json
import csv

from src.models.metrics import compute_metrics, default_metrics

# 从字符串指定路径位置输入类或函数对象
def get_obj_from_str(string, reload=False):
	module, cls = string.rsplit(".", 1)
	module_imp = importlib.import_module(module, package=None) # import指定路径的文件化为对象导入
	if reload:
		importlib.reload(module_imp) # 在运行过程中若修改了库，需要使用reload重新载入
	return getattr(module_imp, cls) # getattr()函数获取对象中对应字符串的对象属性（可以是值、函数等）

# 从配置中载入模型
def instantiate_from_config(config):
	if not "target" in config:
		raise KeyError("Expected key `target` to instantiate.")
	module = get_obj_from_str(config["target"]) # target路径的类或函数模块
	params_config = config.get('params', dict()) # 对应模块的参数配置
	return module(**params_config)

def add_params(config, name:str, value):
	config['params'][name] = value
	return config

# set python hash seed
def set_seed(seed=0):
	os.environ['PYTHONHASHSEED'] = str(seed)
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

def setup_logging(log_file="outputs/training.log"):
	os.makedirs(os.path.dirname(log_file), exist_ok=True)
	# 设置日志输出格式和处理器：同时输出到终端和文件
	logger = logging.getLogger()
	logger.setLevel(logging.INFO)
	formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

	stream_handler = logging.StreamHandler(sys.stdout)
	stream_handler.setFormatter(formatter)

	file_handler = logging.FileHandler(log_file)
	file_handler.setFormatter(formatter)

	logger.handlers.clear()
	logger.addHandler(stream_handler)
	logger.addHandler(file_handler)

# 命令行指令
def parse_args():
	parser = argparse.ArgumentParser(description='Mul-Pro')
	parser.add_argument("-C", "--config", type=str, required=True, help="the path of config file")
	parser.add_argument("--device", type=str, default="0", help="the num of used cuda")
	parser.add_argument("--repeat", type=int, default=1, help="Number of times to repeat the training process")
	parser.add_argument("--root_dir", type=str, default=None, help="subdirectory to save the outputs")
	parser.add_argument("--log", default='standard', help="logging mode, standard, lazy or complete")
	opt, unknown = parser.parse_known_args()
	return opt, unknown

def run_epoch(dataloader, stage='train', metrics_types=None, cal_metrics=True):
	model.train() if stage == 'train' else model.eval()
	
	total_loss = 0.0
	n_total = 0
	all_preds = []
	all_labels = []
	feature_importances = {fea: [] for fea in model.input_encs}

	batch_num = 0
	for data in dataloader:
		data = data.to(device)

		if stage == 'train':
			optimizer.zero_grad()
			loss, out = model.get_loss(data, return_out=True)
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=train_config.get('grad_clip', 10.0))
			optimizer.step()
			scheduler.step()
		else:
			with torch.no_grad():
				loss, out = model.get_loss(data, return_out=True)

		batch_size = data.num_graphs
		total_loss += loss.item() * batch_size
		n_total += batch_size

		batch_num += 1
		if stage == 'train':
			if batch_num % 10 == 0:
				current_lr = optimizer.param_groups[0]['lr']
				dataloader.set_postfix(loss=loss.item(), lr=current_lr)
		else:
			importance = model.get_feature_importance()
			if importance is not None:
				for fea, imp in importance.items():
					feature_importances[fea].append(imp)

		if cal_metrics:
			all_preds.append(out)
			all_labels.append(data.y)

	# calculate average loss
	avg_loss = total_loss / n_total

	# compute metric on full data
	if metrics_types is None:
		metrics_types = [default_metrics(model.task_type)]

	if cal_metrics:
		# concatenate all
		preds_tensor = torch.cat(all_preds, dim=0)
		labels_tensor = torch.cat(all_labels, dim=0)

		metrics = {}
		for metrics_type in metrics_types:	
			print(f"Computing {stage} {metrics_type} for task_type={model.task_type}...")
			metric = compute_metrics(preds_tensor, labels_tensor, model.task_type, metrics_type)
			metrics[metrics_type] = metric
	else:
		metrics = {mt: -1 for mt in metrics_types}

	return avg_loss, metrics, feature_importances

if __name__=="__main__":
	sys.path.append(os.getcwd()) # 将命令本脚本所在文件夹加入环境变量
	set_seed(0)

	opt, unknown = parse_args()
	print(opt)

	#torch.autograd.set_detect_anomaly(True)
	
	# ---------------------
	#  project root config
	# ---------------------
	log_root = os.path.expanduser("./outputs/")
	configs = OmegaConf.load(opt.config)

	# 解析 unknown 里的 "key.subkey=value" 覆盖到 configs
	override_tags = []
	for item in unknown:
		if item.startswith("--"):
			item = item[2:]
		if '=' not in item:
			continue
		key, value = item.split('=', 1)
		# 这里用 OmegaConf.update 支持点路径
		OmegaConf.update(configs, key, OmegaConf.create(value) if value.startswith('{') or value.startswith('[') else value)
		override_tags.append(value)

	config_name = os.path.basename(opt.config).split(".")[0]
	now_time = datetime.datetime.now().strftime("%Y%m%d-T%H-%M") # 训练开始时间
	data_name = configs['data']['train_data']['target'].split('.')[-1]

	# 输出文件夹名包含指定配置
	if len(override_tags) > 0:
		override_str = '-'.join(override_tags)
		dir_name = f"{config_name}-{override_str}-{now_time}"
	else:
		dir_name = f"{config_name}-{now_time}"

	if opt.root_dir is not None:
		project_root = os.path.join(log_root, data_name, opt.root_dir, dir_name)
	else:
		project_root = os.path.join(log_root, data_name, dir_name)

	setup_logging(project_root.rstrip('/') + '_training.log')
	logging.info("The train config has been init:")
	logging.info(OmegaConf.to_yaml(configs))
	print(f"Set the project root at: {project_root}, log mode: {opt.log}")

	# -----------
	#  Load data
	# -----------
	data_config = configs.data
	train_dset = instantiate_from_config(data_config["train_data"])
	valid_dset = instantiate_from_config(data_config["valid_data"])
	if OmegaConf.is_list(data_config["test_data"]):  # 考虑有多个测试集的情况
		test_dset = instantiate_from_config(data_config["test_data"][0])
	else:
		test_dset = instantiate_from_config(data_config["test_data"])

	train_loader = DataLoader(train_dset, batch_size=data_config["batch_size"], shuffle=True, num_workers=data_config["workers"], persistent_workers=True, drop_last=True)
	#train_loader = DataLoader(train_dset, batch_size=data_config["batch_size"], shuffle=True, num_workers=0, drop_last=True)
	#valid_loader = DataLoader(valid_dset, batch_size=data_config["batch_size"], shuffle=False, num_workers=data_config["workers"])
	valid_loader = DataLoader(valid_dset, batch_size=data_config["batch_size"], shuffle=False, num_workers=0)
	#test_loaders = [DataLoader(test_dset, batch_size=data_config["batch_size"], shuffle=False, num_workers=data_config["workers"]) for test_dset in test_dsets]
	test_loader = DataLoader(test_dset, batch_size=data_config["batch_size"], shuffle=False, num_workers=0)

	# model
	if torch.cuda.is_available():
		device = torch.device(f'cuda:{opt.device}')
	else:
		device = torch.device(f'npu:{opt.device}')
	model_config = configs.model
	add_params(model_config, 'num_classes', train_dset.num_classes)

	# -------------
	#  train model
	# -------------
	all_train_metrics, all_valid_metrics, all_test_metrics = {}, {}, {}
	for repeat_i in range(opt.repeat):
		try:
			# model
			model = instantiate_from_config(model_config).to(device)
			model_size = sum(p.numel() for p in model.parameters())
			logging.info("Finished building the model:")
			logging.info("Total params: %.2fM" % (model_size/1e6))

			# training config
			train_config = configs.training
			train_epoch = train_config.epochs
			metrics_types = train_config.get("metrics", None)

			# optimizer config
			learning_rate = train_config.get("lr", 1e-3)
			weight_decay = train_config.get("weight_decay", 1e-6)
			optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

			# learning rate scheduler
			warmup_epochs = train_config.get("warmup_epochs", 0)
			warmup_steps = warmup_epochs * len(train_loader)
			decay_rate = train_config.get('decay_rate', 1.0) # 最终学习率是初始学习率的多少倍
			total_steps = train_epoch * len(train_loader)
			def lr_lambda(current_step: int):
				"""
				Returns a multiplicative factor for the learning rate based on the current step.
				- Linear warmup: from 0 to 1 over `warmup_steps`.
				- Cosine decay: from 1 to decay_rate (10% of max_lr) from warmup_steps to decay_steps.
				"""
				if current_step < warmup_steps:
					return float(current_step) / float(max(1, warmup_steps)) # Linear warmup
				else:
					progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps)) # Cosine decay
					cosine_decay = 0.5 * (1 + math.cos(math.pi * progress)) # cosine decay factor goes from 1 to 0
					return decay_rate + (1.0 - decay_rate) * cosine_decay # Scale to have final lr of decay_rate * max_lr:
			scheduler = LambdaLR(optimizer, lr_lambda)

			begin_time = time.time()
			logging.info(f"Start train the {model_config['target']} model: rest {train_epoch} epochs")

			train_losses, valid_losses, test_losses = [], [], []
			train_metrics_whole, valid_metrics_whole, test_metrics_whole = {}, {}, {}
			feature_importance_list = []

			for epoch in range(train_epoch):
				# 训练阶段
				with tqdm(train_loader, desc=f"Epoch [{epoch+1}/{train_epoch}]") as pbar:
					train_loss, train_metrics, _ = run_epoch(pbar, stage='train', metrics_types=metrics_types, cal_metrics=(epoch>=train_epoch-5)) # 只在最后5个epoch计算指标

				# 验证和测试
				valid_loss, valid_metrics, feature_imp_epoch = run_epoch(valid_loader, stage='valid', metrics_types=metrics_types)
				test_loss, test_metrics, _ = run_epoch(test_loader, stage='test', metrics_types=metrics_types)

				# 记录数据
				train_losses.append(train_loss)
				valid_losses.append(valid_loss)
				test_losses.append(test_loss)
				for key in valid_metrics.keys():
					train_metrics_whole.setdefault(key, []).append(train_metrics[key])
					valid_metrics_whole.setdefault(key, []).append(valid_metrics[key])
					test_metrics_whole.setdefault(key, []).append(test_metrics[key])

				# 打印日志
				logging.info(
					f"Epoch {epoch+1:03d} Learning Rate {optimizer.param_groups[0]['lr']:.2e} | Train Loss: {train_loss:.4f}, " + ', '.join([f'{key}: {value:.4f}' for key, value in train_metrics.items()]) + " | "
					f"Valid Loss: {valid_loss:.4f}, " + ', '.join([f'{key}: {value:.4f}' for key, value in valid_metrics.items()]) + " | "
					f"Test Loss: {test_loss:.4f}, " + ', '.join([f'{key}: {value:.4f}' for key, value in test_metrics.items()])
				)
				
				if (epoch+1) % 10 == 0:
					importance = model.get_feature_importance()
					if importance is not None:
						feature_importance_list.append({fea: np.mean(imp) for fea, imp in feature_imp_epoch.items()})

			running_time = time.time() - begin_time
			logging.info(f"Train time: {running_time//3600}h {running_time%3600//60}m {int(running_time%60)}s")
			for key in valid_metrics_whole.keys():
				train_metrics_list = train_metrics_whole[key]
				valid_metrics_list = valid_metrics_whole[key]
				test_metrics_list = test_metrics_whole[key]
				logging.info(
					f"Final Train Average {key}: mean {np.mean(train_metrics_list[-5:]):.4f}, std {np.std(train_metrics_list[-5:]):.4f}; "
					f"Final Valid Average {key}: mean {np.mean(valid_metrics_list[-5:]):.4f}, std {np.std(valid_metrics_list[-5:]):.4f}; "
					f"Final Test Average {key}: mean {np.mean(test_metrics_list[-5:]):.4f}, std {np.std(test_metrics_list[-5:]):.4f}"
				)
				
				all_train_metrics.setdefault(key, []).extend(train_metrics_list[-5:])
				all_valid_metrics.setdefault(key, []).extend(valid_metrics_list[-5:])
				all_test_metrics.setdefault(key, []).extend(test_metrics_list[-5:])
			logging.info(f"Estimated remaining time: {running_time * (opt.repeat - repeat_i - 1) // 3600}h {running_time * (opt.repeat - repeat_i - 1) % 3600 // 60}m {int(running_time * (opt.repeat - repeat_i - 1) % 60)}s")

			if opt.log != 'lazy':
				os.makedirs(project_root, exist_ok=True)
				OmegaConf.save(configs, os.path.join(project_root, "train.yaml"))
				if opt.log == 'complete':
					torch.save(model.state_dict(), os.path.join(project_root, f"model_{repeat_i}.pth"))

				# 保存曲线图
				plt.figure()
				plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss')
				plt.plot(range(1, len(valid_losses)+1), valid_losses, label='Valid Loss')
				plt.plot(range(1, len(test_losses)+1), test_losses, label='Test Loss')
				plt.xlabel('Epoch')
				plt.ylabel('Loss')
				plt.legend()
				plt.savefig(os.path.join(project_root, f"loss_curve_{repeat_i}.png"))

				plt.figure()
				for key in valid_metrics_whole.keys():
					valid_metrics_list = valid_metrics_whole[key]
					test_metrics_list = test_metrics_whole[key]
					plt.plot(range(1, len(valid_metrics_list)+1), valid_metrics_list, label=f'Valid {key}')
					plt.plot(range(1, len(test_metrics_list)+1), test_metrics_list, label=f'Test {key}')
				plt.xlabel('Epoch')
				plt.ylabel('Metrics')
				plt.legend()
				plt.savefig(os.path.join(project_root, f"metrics_curve_{repeat_i}.png"))

				metrics_save_path = os.path.join(project_root, f"training_logs_{repeat_i}.json")
				with open(metrics_save_path, 'w') as f:
					json.dump({
						"train_loss": train_losses,
						"valid_loss": valid_losses,
						"test_loss": test_losses,
						**valid_metrics_whole,
						**test_metrics_whole,
					}, f, indent=3)

				if feature_importance_list:
					num_epochs = len(feature_importance_list)
					num_features = len(feature_importance_list[0])
					fig, axs = plt.subplots(nrows=(num_epochs + 4) // 5, ncols=min(num_epochs, 5), figsize=(15, 3 * ((num_epochs + 4) // 5)))
					axs = axs.flatten()
					for i, imps in enumerate(feature_importance_list):
						axs[i].bar(imps.keys(), imps.values())
						axs[i].set_title(f"Epoch {i+1}")
						axs[i].tick_params(axis='x', rotation=45)
					plt.tight_layout()
					plt.savefig(os.path.join(project_root, f"feature_importance_over_epochs_{repeat_i}.png"))
					plt.close()

					keys = list(feature_importance_list[0].keys())
					csv_path = os.path.join(project_root, f"feature_importance_{repeat_i}.csv")
					with open(csv_path, 'w', newline='') as f:
						writer = csv.writer(f)
						writer.writerow(['Epoch'] + keys)
						for i, imp in enumerate(feature_importance_list):
							row = [i+1] + [imp.get(k, 0) for k in keys]
							writer.writerow(row)
		except Exception as e:
			logging.error(f"An error occurred during training repeat {repeat_i}: {e}")
			continue
	
	for key in all_valid_metrics.keys():
		all_train_list = all_train_metrics[key]
		all_valid_list = all_valid_metrics[key]
		all_test_list = all_test_metrics[key]
		logging.info(f"All {len(all_train_list)} Train Average {key}: mean {np.mean(all_train_list):.4f}, std {np.std(all_train_list):.4f}; ")
		logging.info(f"All {len(all_valid_list)} Valid Average {key}: mean {np.mean(all_valid_list):.4f}, std {np.std(all_valid_list):.4f}; ")
		logging.info(f"All {len(all_test_list)} Test Average {key}: mean {np.mean(all_test_list):.4f}, std {np.std(all_test_list):.4f}")
