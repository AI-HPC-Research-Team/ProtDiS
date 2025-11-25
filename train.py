import os, time, argparse, logging, sys
import math
import yaml
import importlib
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch.nn.functional as F
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torch
from torch_geometric.loader import DataLoader as PyGLoader
#from torch_npu.optim import NpuFusedAdamW as AdamW
#import torch_npu

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

    if not logger.handlers:
        logger.addHandler(stream_handler)
        logger.addHandler(file_handler)
    else:
        logger.handlers.clear()
        logger.addHandler(stream_handler)
        logger.addHandler(file_handler)

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def load_class(module_path, class_name):
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-C', '--config', type=str, required=True, default="src/configs/config_kon.yaml", help='Path to YAML config file')
    parser.add_argument("-R", "--checkpoint", type=str, help="Path to model checkpoint for resuming training")
    parser.add_argument('--device', type=int, default=0, help='Specify NPU / GPU device ID (default: 0)')
    return parser.parse_args()

def create_output_dir(base_dir="outputs", base_name=''):
    # 使用当前时间生成文件夹名称，例如 _training_20250414_153000
    run_id = time.strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(base_dir, f'{base_name}_training_{run_id}')
    return output_dir

def train(args, config, output_dir):
    train_params = config['training']

    # 初始化日志配置
    log_file = output_dir.rstrip('/') + '.log'
    setup_logging(log_file)
    logging.info("Logging is set up. Log file: {}".format(log_file))
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir="/root/tf-logs")
    except Exception as e:
        logging.warning(f"Could not initialize TensorBoard SummaryWriter: {e}")
        writer = None

    config_str = yaml.dump(config, default_flow_style=False, allow_unicode=True)
    logging.info("Training configuration:\n{}".format(config_str))

    # Load model and dataset classes dynamically
    model_class = load_class(config['model']['module'], config['model']['class'])
    dataset_class = load_class(config['dataset']['module'], config['dataset']['class'])

    model_params = config['model']['params']
    data_params = config['dataset']['params']

    model = model_class(**model_params)
    model_size = sum(p.numel() for p in model.parameters())
    logging.info("Total params: %.2fM" % (model_size/1e6))
    if args.checkpoint:
        logging.info(f"Loading model from checkpoint: {args.checkpoint}")
        state_dict = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(state_dict)

    dataset = dataset_class(**data_params)
    train_loader = DataLoader(dataset, batch_size=train_params['batch_size'], num_workers=8, collate_fn=dataset.collate_fn, shuffle=True)
    #train_loader = PyGLoader(dataset, batch_size=train_params['batch_size'], num_workers=8, shuffle=True)

    # Hyperparameters
    max_lr = float(train_params['max_lr'])
    num_epochs = train_params['num_epochs']
    fix_main_params = train_params.get('fix_main_params', False)
    # learning rate schedule parameters
    warmup_steps = train_params['warmup_steps']
    decay_steps = len(train_loader) * train_params.get('decay_epochs', 500)
    decay_rate = train_params.get('decay_rate', 0.1)
    # adversarial training parameters
    adv_start_epoch = train_params.get('adv_start_epoch', 0)
    adv_warmup_epochs = train_params.get('adv_warmup_epochs', 300)
    adv_lr_scale = train_params.get('adv_lr_scale', 5.0)
    
    total_steps = num_epochs * len(train_loader)
    
    # Optimizer
    adv_params = list(model.adv_branch.parameters()) if hasattr(model, 'adv_branch') else []
    main_params = [p for n, p in model.named_parameters() if ("adv_branch" not in n)]
    if fix_main_params:
        for p in main_params:
            p.requires_grad = False
        logging.info("Fixed main parameters, only training adversarial branch.")
        optimizer = AdamW([
            {'params': adv_params, 'lr': max_lr * adv_lr_scale}
        ], lr=max_lr, betas=(0.9, 0.95), weight_decay=0.01)
    else:
        optimizer = AdamW([
            {'params': main_params, 'lr': max_lr},
            {'params': adv_params, 'lr': max_lr * adv_lr_scale}
        ], lr=max_lr, betas=(0.9, 0.95), weight_decay=0.01)

    # Learning rate scheduler
    def lr_lambda(current_step: int):
        """
        Returns a multiplicative factor for the learning rate based on the current step.
        - Linear warmup: from 0 to 1 over `warmup_steps`.
        - Cosine decay: from 1 to decay_rate (10% of max_lr) from warmup_steps to decay_steps.
        """
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps)) # Linear warmup
        elif current_step < decay_steps:
            progress = float(current_step - warmup_steps) / float(max(1, decay_steps - warmup_steps)) # Cosine decay
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress)) # cosine decay factor goes from 1 to 0
            return decay_rate + (1.0 - decay_rate) * cosine_decay # Scale to have final lr of decay_rate * max_lr:
        else:
            progress = float(current_step - decay_steps) / float(max(1, total_steps - decay_steps)) # Cosine decay
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress)) # cosine decay factor goes from 1 to 0
            return 0.01 + (decay_rate - 0.01) * cosine_decay # Scale to have final lr of 0.01 * max_lr:
    
    scheduler = LambdaLR(optimizer, lr_lambda)

    def get_adv_scale(current_epoch, adv_start_epoch=0, adv_warmup_epochs=300):
        #return min(1.0, min(0, ((current_epoch - adv_start_epoch) / abs(adv_start_epoch))) + max(0, (current_epoch - adv_start_epoch) / adv_warmup_epochs))
        return min(1.0, max(0, (current_epoch - adv_start_epoch) / adv_warmup_epochs))

    # -----------------------------------------
    #  Training Loop
    # -----------------------------------------
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.device}')
    elif torch.npu.is_available():
        device = torch.device(f'npu:{args.device}')
    else:
        device = torch.device('cpu')
    model.to(device)
    logging.info("Using device: {}".format(device))

    global_step = 0
    monitor = train_params.get('monitor', ['ss_loss', 'superfamily_loss'])
    best_monitor_loss = float('inf')
    best_epoch = 0

    losses_by_key = {}
    epoch_monitor_losses = []
    for epoch in range(num_epochs):
        model.train()
        loss_dict = {}
        
        if hasattr(model, 'update_lambda'):
            lambda_adv_scale = get_adv_scale(epoch, adv_start_epoch, adv_warmup_epochs)
            model.update_lambda(lambda_adv_scale)
        current_lambda_adv = model.lambda_adv_local if hasattr(model, 'lambda_adv_local') else 0.0

        with tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch [{epoch+1}/{num_epochs}]") as pbar:
            for batch_idx, batch in pbar:
                batch.to(device)
                optimizer.zero_grad()

                #with torch.autograd.detect_anomaly():
                loss_dict_batch = model(batch)
                total_loss = loss_dict_batch['total_loss']
                total_loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=train_params.get('grad_clip', 5.0))

                optimizer.step()
                scheduler.step()  # update the learning rate

                global_step += 1

                for key, value in loss_dict_batch.items():
                    loss_dict.setdefault(key, 0.0)
                    loss_dict[key] += value.item()

                # Optionally, print training progress
                if global_step % 10 == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    pbar.set_postfix(loss=total_loss.item(), lr=current_lr, lambda_adv=current_lambda_adv)
                
        # After each epoch, print the average loss for the entire epoch
        avg_loss_dict = {key: value / len(train_loader) for key, value in loss_dict.items()}
        for key, value in avg_loss_dict.items():
            losses_by_key.setdefault(key, [])
            losses_by_key[key].append(value)

        monitor_loss = sum(avg_loss_dict.get(loss, 0.0) for loss in monitor)
        epoch_monitor_losses.append(monitor_loss)

        if writer:
            for key, value in avg_loss_dict.items():
                writer.add_scalar(f'Loss/{key}', value, epoch+1)
            writer.add_scalar('Monitor Loss', monitor_loss, epoch+1)

        loss_details = " | ".join([f"{key}: {value:.4f}" for key, value in avg_loss_dict.items()])
        logging.info(f"Epoch [{epoch+1}/{num_epochs}] Learning Rate {current_lr:.2e} Lambda Adv: {current_lambda_adv:.3f} | {loss_details} | Monitor Loss: {monitor_loss:.4f}")

        if monitor_loss < best_monitor_loss:
            best_monitor_loss = monitor_loss
            best_epoch = epoch + 1
            best_model_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            logging.info(f"New best model found at epoch {best_epoch} with monitor_loss {best_monitor_loss:.4f}")

        # Save the model state every 50 epochs
        if (epoch + 1) % 50 == 0:
            os.makedirs(output_dir, exist_ok=True)
            model_save_path = os.path.join(output_dir, f'model_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), model_save_path)
            logging.info(f"Model saved at {model_save_path}")

    # -----------------------------------------
    #  Save the model and plot loss curves
    # -----------------------------------------    
    logging.info("Training complete.")

    os.makedirs(output_dir, exist_ok=True)
    config_save_path = os.path.join(output_dir, os.path.basename(train_params.get('config_save_path', 'config.yaml')))
    model_save_path = os.path.join(output_dir, os.path.basename(train_params.get('model_save_path', f'model_ep{best_epoch}.pth')))
    loss_curve_path = os.path.join(output_dir, os.path.basename(train_params.get('loss_curve_path', 'loss_curve.png')))

    # 保存配置文件
    with open(config_save_path, 'w') as file:
        yaml.dump(config, file)
    # 保存最优模型
    torch.save(best_model_state_dict, model_save_path)
    logging.info(f"Model at epoch {best_epoch} saved to {model_save_path} with monitor_loss {best_monitor_loss:.4f}")

    fig, axs = plt.subplots(2, 1, figsize=(10, 12))
    epochs = range(1, num_epochs + 1)
    # 第一幅图：绘制 avg_loss_dict 中所有 loss 曲线
    for key, loss_list in losses_by_key.items():
        if key != 'total_loss':
            axs[0].plot(epochs, loss_list, label=key)
    axs[0].set_title('Average Loss per Loss Type (avg_loss_dict)')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    axs[0].grid(True)

    # 第二幅图：绘制 monitor_loss 曲线
    axs[1].plot(epochs, epoch_monitor_losses, color='red', label='Monitor Loss')
    axs[1].set_title('Monitor Loss Curve')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Monitor Loss')
    axs[1].legend()
    axs[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(loss_curve_path)
    plt.close()
    logging.info(f"Loss curve saved to {loss_curve_path}")

if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    output_dir = create_output_dir(base_name=os.path.basename(args.config)[:-5])
    train(args, config, output_dir)
