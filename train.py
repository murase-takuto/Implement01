import json
import os

import numpy as np

import misc.utils as utils
import opts
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_value_
from dataloader import VideoDataset
from misc.rewards import get_self_critical_reward, init_cider_scorer
from models import DecoderRNN, EncoderRNN, S2VTAttModel, S2VTModel
from torch import nn
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter

def train(loader, model, crit, optimizer, lr_scheduler, opt, rl_crit=None):
    """
    学習する。
    
    Parameters
    ----------
    loader : int
        対象の果物のマスタID。
    model : int
        対象地域のマスタID。
    potimizer : int
        対象地域のマスタID。
    lr_scheduler : int
        対象地域のマスタID。
    model : int
        対象地域のマスタID。
    opt : int
        対象地域のマスタID。
    rl_crit : int, default=None
        対象地域のマスタID。

    Returns
    -------
    consumption_tax : int
        消費税値。
    """

    # tensorboard用
    writer = SummaryWriter("./train_log/20220628-LSTM")
    iterations = 0

    model.train()
    #model = nn.DataParallel(model)
    for epoch in range(opt["epochs"]):
        lr_scheduler.step()

        iteration = 0
        # If start self crit training
        if opt["self_crit_after"] != -1 and epoch >= opt["self_crit_after"]:
            sc_flag = True
            init_cider_scorer(opt["cached_tokens"])
        else:
            sc_flag = False

        for data in loader:
            torch.cuda.synchronize()
            fc_feats = data['fc_feats'].cuda()
            labels = data['labels'].cuda()
            masks = data['masks'].cuda()

            optimizer.zero_grad()
            if not sc_flag:
                seq_probs, _ = model(fc_feats, labels, 'train')
                loss = crit(seq_probs, labels[:, 1:], masks[:, 1:])
            else:
                seq_probs, seq_preds = model(fc_feats, mode='inference', opt=opt)
                reward = get_self_critical_reward(model, fc_feats, data, seq_preds)
                print(reward.shape)
                loss = rl_crit(seq_probs, seq_preds, torch.from_numpy(reward).float().cuda())

            loss.backward()
            clip_grad_value_(model.parameters(), opt['grad_clip'])
            optimizer.step()
            train_loss = loss.item()
            torch.cuda.synchronize()
            iteration += 1

            if not sc_flag:
                print("iter %d (epoch %d / %d), train_loss = %.6f" % (iteration, epoch, opt["epochs"], train_loss))
            else:
                print("iter %d (epoch %d / %d), avg_reward = %.6f" % (iteration, epoch, opt["epochs"], np.mean(reward[:, 0])))
            
            # 損失関数の値をtensorboardに記録
            writer.add_scalar("TRAIN_LOSS", train_loss, iterations)
            iterations += 1

        # if epoch % opt["save_checkpoint_every"] == 0:
        checkpoint_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
            11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
            21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
            31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
            41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
            60, 70, 80, 90, 100, 200, 300, 400, 500,
            1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
        if epoch in checkpoint_list:
            model_path = os.path.join(opt["checkpoint_path"], 'model_%d.pth' % (epoch))
            model_info_path = os.path.join(opt["checkpoint_path"], 'model_score.txt')
            torch.save(model.state_dict(), model_path)
            print("model saved to %s" % (model_path))
            with open(model_info_path, 'a') as f:
                f.write("model_%d, loss: %.6f\n" % (epoch, train_loss))
        
    writer.close()


def main(opt):
    dataset = VideoDataset(opt, 'train')
    dataloader = DataLoader(dataset, batch_size=opt["batch_size"], shuffle=True)
    opt["vocab_size"] = dataset.get_vocab_size()
    if opt["model"] == 'S2VTModel':
        model = S2VTModel(
            opt["vocab_size"],
            opt["max_len"],
            opt["dim_hidden"],
            opt["dim_word"],
            opt['dim_vid'],
            rnn_cell=opt['rnn_type'],
            n_layers=opt['num_layers'],
            rnn_dropout_p=opt["rnn_dropout_p"])
    elif opt["model"] == "S2VTAttModel":
        encoder = EncoderRNN(
            opt["dim_vid"],
            opt["dim_hidden"],
            bidirectional=opt["bidirectional"],
            input_dropout_p=opt["input_dropout_p"],
            rnn_cell=opt['rnn_type'],
            rnn_dropout_p=opt["rnn_dropout_p"])
        decoder = DecoderRNN(
            opt["vocab_size"],
            opt["max_len"],
            opt["dim_hidden"],
            opt["dim_word"],
            input_dropout_p=opt["input_dropout_p"],
            rnn_cell=opt['rnn_type'],
            rnn_dropout_p=opt["rnn_dropout_p"],
            bidirectional=opt["bidirectional"])
        model = S2VTAttModel(encoder, decoder)
    model = model.cuda()
    crit = utils.LanguageModelCriterion()
    rl_crit = utils.RewardCriterion()
    optimizer = optim.Adam(
        model.parameters(),
        lr=opt["learning_rate"],
        weight_decay=opt["weight_decay"])
    exp_lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=opt["learning_rate_decay_every"],
        gamma=opt["learning_rate_decay_rate"])

    train(dataloader, model, crit, optimizer, exp_lr_scheduler, opt, rl_crit)


if __name__ == '__main__':
    # コマンドラインからの引数管理
    opt = opts.parse_opt()
    opt = vars(opt)
    # os.environ['CUDA_VISIBLE_DEVICES'] = opt["gpu"]
    opt_json = os.path.join(opt["checkpoint_path"], 'opt_info.json')
    if not os.path.isdir(opt["checkpoint_path"]):
        os.mkdir(opt["checkpoint_path"])
    with open(opt_json, 'w') as f:
        json.dump(opt, f)
    print('save opt details to %s' % (opt_json))
    main(opt)
