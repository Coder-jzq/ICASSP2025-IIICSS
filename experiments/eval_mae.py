import argparse
import os

import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.tools import get_variance_level, to_device, log, synth_one_sample
from dataset import Dataset

from experiments.maeloss import MAELoss

from models import CompTransTTS, ScheduledOptim

from utils.tools import get_configs_of

rank = 4
device = torch.device('cuda:{}'.format(rank) if torch.cuda.is_available() else 'cpu')

def get_model_test(ckpt_path, configs):
    (preprocess_config, model_config, train_config) = configs

    model = CompTransTTS(preprocess_config, model_config, train_config, False).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["models"])

    model.eval()
    model.requires_grad_ = False
    return model

def evaluate_test(model, configs):
    preprocess_config, model_config, train_config = configs

    # Get dataset
    level_tag, *_ = get_variance_level(preprocess_config, model_config)
    dataset = Dataset(
        "test_{}.txt".format(level_tag), preprocess_config, model_config, train_config, sort=False, drop_last=True
    )
    batch_size = train_config["optimizer"]["batch_size"]
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
    )


    # Get loss function
    Loss = MAELoss(preprocess_config, model_config, train_config).to(device)

    # 存储和
    losses_end = [0.0, 0.0, 0.0, 0.0, 0.0]

    # 存储步骤
    step = 0

    for batchs in loader:
        for batch in batchs:
            batch = to_device(batch, device)
            with torch.no_grad():
                # Forward
                output, cov = model(*(batch[2:]), step= 400000)
                batch[9:11], output = output[-2:], output[:-2] # Update pitch and energy level

                # Cal Loss
                losses = Loss(batch, output)

                losses_end[0] = losses_end[0] + losses[0]
                losses_end[1] = losses_end[1] + losses[1]
                losses_end[2] = losses_end[2] + losses[2]
                losses_end[3] = losses_end[3] + losses[3]
                losses_end[4] = losses_end[4] + losses[4]

                step = step + 1


    message = "Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Pitch Loss: {:.4f}, Energy Loss: {:.4f}, Duration Loss: {:.4f}".format(
        *(losses_end)
    )
    # 平均情况
    losses_end_avg = [x / step for x in losses_end]

    message2 = "Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Pitch Loss: {:.4f}, Energy Loss: {:.4f}, Duration Loss: {:.4f}".format(
        *(losses_end_avg)
    )
    # print(message)
    # print("step:",step)
    print(message2)


def get_configs_of_test(dataset):
    config_dir = os.path.join("../config", dataset)
    preprocess_config = yaml.load(open(
        os.path.join(config_dir, "preprocess.yaml"), "r"), Loader=yaml.FullLoader)
    model_config = yaml.load(open(
        os.path.join(config_dir, "model.yaml"), "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(
        os.path.join(config_dir, "train.yaml"), "r"), Loader=yaml.FullLoader)
    return preprocess_config, model_config, train_config


if __name__ == '__main__':
    # 获取配置
    preprocess_config, model_config, train_config = get_configs_of_test("DailyTalk")
    configs = (preprocess_config, model_config, train_config)

    # 获取模型, 注意输入ckpt的路径
    model = get_model_test("/xxx/ckpt/DailyTalk/400000.pth.tar", configs)

    evaluate_test(model, configs)
