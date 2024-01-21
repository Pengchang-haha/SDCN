"""
PyTorch 1.7 implementation of the following paper:
    @inproceedings{RADN2021ntire,
    title={Region-Adaptive Deformable Network for Image Quality Assessment},
    author={Shuwei Shi and Qingyan Bai and Mingdeng Cao and Weihao Xia and Jiahao Wang and Yifan Chen and Yujiu Yang},
    booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops},
    year={2021}
    }

 Requirements: See requirements.txt.
    ```bash
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

 Acknowledgments: The codes are based on WaDIQaM and we really appreciate it.

 Implemented by Qingyan Bai, Shuwei Shi
 Email: baiqingyan1998@gamil.com, ssw20@mails.tsinghua.edu.cn
 Date: 2021/5/7
"""

# -*- coding : utf-8 -*-

from argparse import ArgumentParser
import h5py
import os

import random
from scipy import stats

from PIL import Image
from time import strftime, localtime
import glob
# import scipy
# from scipy import io

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.transforms.functional import to_tensor

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics.metric import Metric

from model.SDSA import *
# from model.one_deformable.SDSA_one_deformable import *
# from model.two_deformable.SDSA_two_deformable import *
from PIL import Image, ImageDraw
from draw_util import *
#from model.WResNet import *


def default_loader(path, channel=3):
    """
    :param path: image path
    :param channel: # image channel
    :return: image
    """
    if channel == 1:
        return Image.open(path).convert('L')
    else:
        assert (channel == 3)
        return Image.open(path).convert('RGB')  #


def RandomCropPatches(im, ref=None, patch_size=32, n_patches=32):  # 32   32 或 224   32
    """
    Random Crop Patches
    :param im: the distorted image
    :param ref: the reference image if FR-IQA is considered (default: None)
    :param patch_size: patch size (default: 32)
    :param n_patches: numbers of patches (default: 32)
    :return: patches
    """
    w, h = im.size

    patches = ()
    ref_patches = ()
    for i in range(n_patches):
        w1 = np.random.randint(low=0, high=w - patch_size + 1)
        h1 = np.random.randint(low=0, high=h - patch_size + 1)
        patch = to_tensor(im.crop((w1, h1, w1 + patch_size, h1 + patch_size)))
        patches = patches + (patch,)
        if ref is not None:
            ref_patch = to_tensor(ref.crop((w1, h1, w1 + patch_size, h1 + patch_size)))
            ref_patches = ref_patches + (ref_patch,)
    if ref is not None:
        return torch.stack(patches), torch.stack(ref_patches)
    else:
        return torch.stack(patches)


def NonOverlappingCropPatches(im, ref=None, patch_size=32):  # 32  224
    """
    NonOverlapping Crop Patches
    :param im: the distorted image
    :param ref: the reference image if FR-IQA is considered (default: None)
    :param patch_size: patch size (default: 32)
    :return: patches
    """
    w, h = im.size

    patches = ()
    ref_patches = ()
    stride = patch_size

    if w % stride == 0 and h % stride == 0:
        i_end = h
        j_end = w
    else:
        i_end = h - stride
        j_end = w - stride

    for i in range(0, i_end, stride):
        for j in range(0, j_end, stride):
            patch = to_tensor(im.crop((j, i, j + patch_size, i + patch_size)))
            patches = patches + (patch,)
            if ref is not None:
                ref_patch = to_tensor(ref.crop((j, i, j + patch_size, i + patch_size)))
                ref_patches = ref_patches + (ref_patch,)

    if ref is not None:
        return torch.stack(patches), torch.stack(ref_patches)
    else:
        return torch.stack(patches)


class IQADataset_less_memory(Dataset):
    """
    IQA Dataset (less memory) - mainly for training
    """

    def __init__(self, args, status='train', loader=default_loader, less_data=False):
        """
        :param args:
        :param status: train/val/test
        :param loader: image loader
        """
        self.status = status
        self.patch_size = args.patch_size
        self.n_patches = args.n_patches
        self.loader = loader

        Info = h5py.File(args.data_info, 'r')
        index = Info['index']
        if less_data:
            index = index[:, args.exp_id % index.shape[1]][:5]
        else:
            index = index[:, args.exp_id % index.shape[1]]
        ref_ids = Info['ref_ids'][0, :]  #

        K = args.K_fold
        k = args.k_test

        valindex = index[int((k - 1) / K * len(index)):int(k / K * len(index))]
        testindex = valindex
        trainindex = [i for i in index if i not in valindex]

        train_index, val_index, test_index = [], [], []
        print(trainindex, valindex, testindex)
        for i in range(len(ref_ids)):
            if ref_ids[i] in trainindex:
                train_index.append(i)
            if ref_ids[i] in testindex:
                test_index.append(i)
            if ref_ids[i] in valindex:
                val_index.append(i)
        if 'train' in status:
            self.index = train_index
            print("# Train Images: {}".format(len(self.index)))
        if 'test' in status:
            self.index = test_index
            print("# Test Images: {}".format(len(self.index)))
        if 'val' in status:
            self.index = val_index
            print("# Val Images: {}".format(len(self.index)))

        self.scale = Info['subjective_scores'][0, :].max()
        self.mos = Info['subjective_scores'][0, self.index] / self.scale
        self.mos_std = Info['subjective_scoresSTD'][0, self.index] / self.scale
        im_names = [Info[Info['image_name'][0, :][i]][()].tobytes()[::2].decode() for i in self.index]
        ref_names = [Info[Info['ref_names'][0, :][i]][()].tobytes()[::2].decode()
                     for i in (ref_ids[self.index] - 1).astype(int)]

        self.patches = ()
        self.label = []
        self.label_std = []
        self.im_names = []
        self.ref_names = []
        # 这里为啥不是for idx in self.index,如果直接取0-len(index)直觉上认为会出问题的，一般是取index中的每个元素出来读取图片
        for idx in range(len(self.index)):
            self.im_names.append(os.path.join(args.im_dir, im_names[idx]))
            if args.ref_dir is None or 'NR' in args.model:
                self.ref_names.append(None)
            else:
                self.ref_names.append(os.path.join(args.ref_dir, ref_names[idx]))

            self.label.append(self.mos[idx])
            self.label_std.append(self.mos_std[idx])

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        im = self.loader(self.im_names[idx])
        if self.ref_names[idx] is None:
            ref = None
        else:
            ref = self.loader(self.ref_names[idx])

        if self.status == 'train':
            patches = RandomCropPatches(im, ref, self.patch_size, self.n_patches)
        else:
            # 考虑为NonOverlappingCropPatches指定n_patches
            # patches = NonOverlappingCropPatches(im, ref, self.patch_size, self.n_patches)
            patches = NonOverlappingCropPatches(im, ref, self.patch_size)
        return patches, (torch.Tensor([self.label[idx], ]), torch.Tensor([self.label_std[idx], ]))


class IQADataset(Dataset):
    """
    IQA Dataset - mainly for validating and testing
    """

    def __init__(self, args, status='train', loader=default_loader):
        """
        :param args:
        :param status: train/val/test
        :param loader: image loader
        """
        self.status = status
        self.patch_size = args.patch_size
        self.n_patches = args.n_patches

        Info = h5py.File(args.data_info, 'r')
        index = Info['index']
        index = index[:, args.exp_id % index.shape[1]]
        ref_ids = Info['ref_ids'][0, :]  #

        K = args.K_fold
        k = args.k_test

        valindex = index[int((k - 1) / K * len(index)):int(k / K * len(index))]
        testindex = valindex
        trainindex = [i for i in index if i not in valindex]
        train_index, val_index, test_index = [], [], []
        # print(trainindex, valindex, testindex)

        for i in range(len(ref_ids)):
            if ref_ids[i] in trainindex:
                train_index.append(i)
            if ref_ids[i] in testindex:
                test_index.append(i)
            if ref_ids[i] in valindex:
                val_index.append(i)
        if 'train' in status:
            self.index = train_index
            print("# Train Images: {}".format(len(self.index)))
        if 'test' in status:
            self.index = test_index
            print("# Test Images: {}".format(len(self.index)))
        if 'val' in status:
            self.index = val_index
            print("# Val Images: {}".format(len(self.index)))


        self.scale = Info['subjective_scores'][0, :].max()
        self.mos = Info['subjective_scores'][0, self.index] / self.scale  #
        self.mos_std = Info['subjective_scoresSTD'][0, self.index] / self.scale
        im_names = [Info[Info['image_name'][0, :][i]][()].tobytes()[::2].decode() for i in self.index]
        ref_names = [Info[Info['ref_names'][0, :][i]][()].tobytes()[::2].decode()
                     for i in (ref_ids[self.index] - 1).astype(int)]

        self.patches = ()
        self.paint_patches = ()  # add
        self.label = []
        self.label_std = []
        self.ims = []
        self.refs = []

        if args.paint_test:
            # 用于可视化图片测试
            image_files = glob.glob(os.path.join(args.paint_dir, '*.jpg')) + glob.glob(
                os.path.join(args.paint_dir, '*.jpeg')) + glob.glob(os.path.join(args.paint_dir, '*.png')) + glob.glob(os.path.join(args.paint_dir, '*.bmp'))
            ref_files = glob.glob(os.path.join(args.paint_ref_dir, '*.bmp'))
            self.paint_im_list = [os.path.basename(file) for file in image_files]
            self.paint_ref_list = [os.path.basename(file) for file in ref_files]
            for index in range(len(self.paint_im_list)):
                img_name = self.paint_im_list[index]
                ref_name = self.paint_ref_list[index]
                im = loader(os.path.join(args.paint_dir, img_name))
                args.ori_imgs.append(im)
                ref = loader(os.path.join(args.paint_ref_dir, ref_name))
                patches = NonOverlappingCropPatches(im, ref, args.patch_size)
                self.paint_patches = self.paint_patches + (patches,)


        for idx in range(len(self.index)):
            im = loader(os.path.join(args.im_dir, im_names[idx]))
            if args.ref_dir is None or 'NR' in args.model:
                ref = None
            else:
                ref = loader(os.path.join(args.ref_dir, ref_names[idx]))

            self.label.append(self.mos[idx])
            self.label_std.append(self.mos_std[idx])

            if status == 'train':
                self.ims.append(im)
                self.refs.append(ref)
            elif status == 'test' or status == 'val':
                patches = NonOverlappingCropPatches(im, ref, args.patch_size)
                self.patches = self.patches + (patches,)

    def __len__(self):
        return len(self.index) if not args.paint_test else len(self.paint_patches)

    def __getitem__(self, idx):
        if self.status == 'train':
            patches = RandomCropPatches(self.ims[idx], self.refs[idx], self.patch_size, self.n_patches)
        else:
            patches = self.patches[idx] if not args.paint_test else self.paint_patches[idx]
        return patches, (torch.Tensor([self.label[idx], ]), torch.Tensor([self.label_std[idx], ]))


def mkdirs(path):
    os.makedirs(path, exist_ok=True)


class IQALoss(torch.nn.Module):
    def __init__(self):
        super(IQALoss, self).__init__()

    def forward(self, y_pred, y):
        """
        loss function, e.g., l1 loss
        :param y_pred: predicted values
        :param y: y[0] is the ground truth label
        :return: the calculated loss
        """
        y_pred = y_pred  # tensor shape[bs, 1]
        y_gt = y[0]  # [bs, 1]
        diff = y_gt - y_pred
        loss = torch.mean(diff * diff)
        return loss


class IQAPerformance(Metric):
    """
    Evaluation of IQA methods using SROCC, KROCC, PLCC, RMSE, MAE.
    `update` must receive output of the form (y_pred, y).
    """

    def reset(self):
        self._y_pred = []
        self._y = []
        self._y_std = []

    def update(self, output):
        y_pred, y = output
        self._y.append(y[0].item())
        self._y_std.append(y[1].item())
        n = int(y_pred.size(0) / y[0].size(0))  # n=1 if images; n>1 if patches
        y_pred_im = y_pred.reshape((y[0].size(0), n)).mean(dim=1, keepdim=True)
        self._y_pred.append(y_pred_im.item())

    def compute(self):
        sq = np.reshape(np.asarray(self._y), (-1,))
        sq_std = np.reshape(np.asarray(self._y_std), (-1,))
        q = np.reshape(np.asarray(self._y_pred), (-1,))

        srocc = stats.spearmanr(sq, q)[0]
        krocc = stats.stats.kendalltau(sq, q)[0]
        plcc = stats.pearsonr(sq, q)[0]
        rmse = np.sqrt(((sq - q) ** 2).mean())
        mae = np.abs((sq - q)).mean()
        outlier_ratio = (np.abs(sq - q) > 2 * sq_std).mean()

        return srocc, krocc, plcc, rmse, mae, outlier_ratio


def get_data_loaders(args):
    """ Prepare the train-val-test data
    :param args: related arguments
    :return: train_loader, val_loader, test_loader, scale
    """
    train_dataset = IQADataset_less_memory(args, 'train', less_data=args.use_less_data)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=4)  # num_workers=4

    if args.use_less_data:
        val_dataset = IQADataset_less_memory(args, 'val', less_data=args.use_less_data)
        test_dataset = IQADataset_less_memory(args, 'test', less_data=args.use_less_data)
    else:
        val_dataset = IQADataset(args, 'val')
        test_dataset = IQADataset(args, 'test')
    val_loader = torch.utils.data.DataLoader(val_dataset)

    test_loader = torch.utils.data.DataLoader(test_dataset)

    scale = test_dataset.scale

    return train_loader, val_loader, test_loader, scale


def run(args):
    """
    Run the program
    """
    train_loader, val_loader, test_loader, scale = get_data_loaders(args)


    # device = torch.device("cuda" if not args.disable_gpu and torch.cuda.is_available() else "cpu")

    device = torch.device("cuda")
    lr_ratio = 1

    # Model instantiation模型实例化

    if args.model == 'WResNet':
        model = WResNet(weighted_average=args.weighted_average)
        # model = OriginalWResNet(weighted_average=args.weighted_average)
    elif args.model == 'SDSA':
        model = SDSA(args, weighted_average=args.weighted_average)
        if args.resume is not None:
            model.load_state_dict(torch.load(args.resume))
    else:
        print('Wrong model name!')
    '''
    if args.model == 'SDSA':
        model = SDSA(weighted_average=args.weighted_average)
    elif args.model == 'WResNet':
        model = WResNet(weighted_average=args.weighted_average)
        if args.resume is not None:
            model.load_state_dict(torch.load(args.resume))
    else:
        print('Wrong model name!')
'''
    # Summary
    writer = SummaryWriter(log_dir=args.log_dir)
    model = model.to(device)
    print(model)

    # Multi-GPU processing
    if args.multi_gpu and torch.cuda.device_count() > 1:
        print("Using multiple GPU")
        model = nn.DataParallel(model)
        # batch_size becomes batch_size * torch.cuda.device_count()

        all_params = model.module.parameters()
        regression_params = []
        for pname, p in model.module.named_parameters():
            if pname.find('fc') >= 0:
                regression_params.append(p)
        regression_params_id = list(map(id, regression_params))
        features_params = list(filter(lambda p: id(p) not in regression_params_id, all_params))
        optimizer = Adam([{'params': regression_params},
                          {'params': features_params, 'lr': args.lr * lr_ratio}],
                         lr=args.lr, weight_decay=args.weight_decay)
    else:
        all_params = model.parameters()
        regression_params = []
        for pname, p in model.named_parameters():
            if pname.find('fc') >= 0:
                regression_params.append(p)
        regression_params_id = list(map(id, regression_params))
        features_params = list(filter(lambda p: id(p) not in regression_params_id, all_params))
        optimizer = Adam([{'params': regression_params},
                          {'params': features_params, 'lr': args.lr * lr_ratio}],
                         lr=args.lr, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.decay_interval, gamma=args.decay_ratio)

    global best_criterion
    global cc_criterion
    best_criterion = 9999  # RMSE
    cc_criterion = -1  # SROCC >= -1

    # trainer
    trainer = create_supervised_trainer(model, optimizer, IQALoss(), device=device)
    evaluator = create_supervised_evaluator(model,
                                            metrics={'IQA_performance': IQAPerformance()},
                                            device=device)

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        loss = engine.state.output
        # writer.add_scalar("training/loss", scale * engine.state.output, engine.state.iteration)
        writer.add_scalar("training/loss", loss, engine.state.iteration)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        SROCC, KROCC, PLCC, RMSE, MAE, OR = metrics['IQA_performance']
        print(strftime("%Y-%m-%d %H:%M:%S", localtime()))
        info = "Validation Results - Epoch: {} SROCC: {:.4f} KROCC: {:.4f} PLCC: {:.4f} RMSE: {:.4f} MAE: {:.4f} OR: {:.2f}%" \
            .format(engine.state.epoch, SROCC, KROCC, PLCC, scale * RMSE, scale * MAE, 100 * OR)
        print(info)

        writer.add_scalar("SROCC/validation", SROCC, engine.state.epoch)
        writer.add_scalar("KROCC/validation", KROCC, engine.state.epoch)
        writer.add_scalar("PLCC/validation", PLCC, engine.state.epoch)
        writer.add_scalar("RMSE/validation", scale * RMSE, engine.state.epoch)
        writer.add_scalar("MAE/validation", scale * MAE, engine.state.epoch)
        writer.add_scalar("OR/validation", OR, engine.state.epoch)

        scheduler.step(engine.state.epoch)  # 在scheduler的step_size表示scheduler.step()每调用step_size次，
        # 对应的学习率就会按照策略调整一次。
        curlr = optimizer.state_dict()['param_groups'][0]['lr']
        print('Current lr: {}'.format(curlr))

        RMSEshow = scale * RMSE

        modelSaveName = 'epcheckpoints/{}-{}-{}-lr={}-bs={}-{:.5f}-{:.5f}-{:.5f}-ep{}'.format(args.model, args.k_test,
                                                                                              args.K_fold, args.lr,
                                                                                              args.batch_size, RMSEshow,
                                                                                              SROCC, PLCC,
                                                                                              engine.state.epoch)
        modelSaveName2 = 'dwcheckpoints/{}-{}-{}-lr={}-bs={}-{:.5f}-{:.5f}-{:.5f}-ep{}'.format(args.model, args.k_test,
                                                                                               args.K_fold, args.lr,
                                                                                               args.batch_size,
                                                                                               RMSEshow,
                                                                                               SROCC, PLCC,
                                                                                               engine.state.epoch)
        args.trained_model_file = modelSaveName2
        # save checkpoints every 20 epochs
        if engine.state.epoch % 20 == 0:  # 20
            try:
                # torch.save(model.module.state_dict(), modelSaveName, _use_new_zipfile_serialization=False)
                torch.save(model.module.state_dict(), args.trained_model_file)
            except:
                # torch.save(model.state_dict(), modelSaveName, _use_new_zipfile_serialization=False)
                torch.save(model.state_dict(), args.trained_model_file)
        global best_criterion
        global best_epoch
        global cc_criterion

        # save checkpoints performing better on RMSE
        if RMSE < best_criterion and engine.state.epoch / args.epochs > 1 / 50:  # 1/50
            best_criterion = RMSE
            best_epoch = engine.state.epoch
            try:
                # torch.save(model.module.state_dict(), modelSaveName2, _use_new_zipfile_serialization=False)
                torch.save(model.module.state_dict(), args.trained_model_file + "-best")
            except Exception as e:
                # torch.save(model.state_dict(), modelSaveName2, _use_new_zipfile_serialization=False)
                torch.save(model.state_dict(), args.trained_model_file + "-best")

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_testing_results(engine):
        if args.test_during_training:
            evaluator.run(test_loader)
            metrics = evaluator.state.metrics
            SROCC, KROCC, PLCC, RMSE, MAE, OR = metrics['IQA_performance']
            print(
                "Testing Results    - Epoch: {} SROCC: {:.4f} KROCC: {:.4f} PLCC: {:.4f} RMSE: {:.4f} MAE: {:.4f} OR: {:.2f}%"
                .format(engine.state.epoch, SROCC, KROCC, PLCC, scale * RMSE, scale * MAE, 100 * OR))
            writer.add_scalar("SROCC/testing", SROCC, engine.state.epoch)
            writer.add_scalar("KROCC/testing", KROCC, engine.state.epoch)
            writer.add_scalar("PLCC/testing", PLCC, engine.state.epoch)
            writer.add_scalar("RMSE/testing", scale * RMSE, engine.state.epoch)
            writer.add_scalar("MAE/testing", scale * MAE, engine.state.epoch)
            writer.add_scalar("OR/testing", OR, engine.state.epoch)

    @trainer.on(Events.COMPLETED)
    def final_testing_results(engine):
        global best_epoch
        model.load_state_dict(torch.load(args.trained_model_file))
        evaluator.run(test_loader)
        metrics = evaluator.state.metrics
        SROCC, KROCC, PLCC, RMSE, MAE, OR = metrics['IQA_performance']
        print(
            "Final Test Results - Epoch: {} SROCC: {:.4f} KROCC: {:.4f} PLCC: {:.4f} RMSE: {:.4f} MAE: {:.4f} OR: {:.2f}%"
            .format(best_epoch, SROCC, KROCC, PLCC, scale * RMSE, scale * MAE, 100 * OR))
        np.save(args.save_result_file, (SROCC, KROCC, PLCC, scale * RMSE, scale * MAE, OR))

    # 这里是为可视化添加的代码，开启paint_test选项之后才会运行
    if args.paint_test:
        evaluator.run(test_loader)
        # 后面的是哪个参数分别表示，取用哪个patch，patch中可视化哪个位置，选择哪一张图片，具体内容参考注释
        # only_paint_samplePoint(args, test_loader, (2, 8), (6, 4), 0,
        #                        deformable_num=2, show_img=True)

        mean_paint_lists = [[np.mean(paint_mat, axis=1).reshape(9, 9) for paint_mat in paint_list] for paint_list in
                            args.paint_lists[0:5]]
        for choose in [0, 1, 2, 3, 4]:
            heatmap_dir = "./visualize_data/heatmap/with_attention"
            if not os.path.exists(heatmap_dir):
                os.makedirs(heatmap_dir)
            choose_paint(mean_paint_lists, args, test_loader, heatmap_dir, choose)
        print("finish paint!")
    else:
        # kick everything off
        trainer.run(train_loader, max_epochs=args.epochs)

    writer.close()


if __name__ == "__main__":
    parser = ArgumentParser(description='PyTorch SDSA')  # SDSA
    parser.add_argument("--seed", type=int, default=19920517)
    # training parameters
    parser.add_argument('--lr', type=float, default=0.00005,
                        help='learning rate (default: 1e-4)')  # 1e-4  0.0001
    parser.add_argument('--batch_size', type=int, default=4,
                        help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs to train')  # 200
    parser.add_argument('--decay_interval', type=int, default=100,#50
                        help='learning rate decay interval')
    parser.add_argument('--decay_ratio', type=float, default=0.8,
                        help='learning rate decay ratio')

    parser.add_argument('--exp_id', default=0, type=int,
                        help='exp id for train-val-test splits')
    parser.add_argument('--K_fold', type=int, default=10,
                        help='K-fold cross-validation')
    parser.add_argument('--k_test', type=int, default=10,
                        help='The k-th fold used for test')

    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='weight decay (default: 0.0)')
    parser.add_argument('--resume',
                        default="D:/1pengchang/project_deformable/dwcheckpoints/SDSA-10-10-lr=0.0001-bs=2-69.96517-0.78542-0.81516-ep151-best",
                        # default="/home/cyl/文档/work/daimai可视化/model/one_deformable/SDSA-10-10-lr=8e-05-bs=2-71.15027-0.76700-0.80866-ep180",
                        # default="/home/cyl/文档/work/daimai可视化/model/two_deformable/SDSA-10-10-lr=1e-05-bs=2-73.24872-0.76984-0.79620-ep140",
                        type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument("--log_dir", type=str, default="tensorboard_logs",
                        help="log directory for Tensorboard log output")
    parser.add_argument("--disable_visualization", action='store_true',
                        help='flag whether to disable TensorBoard visualization')
    parser.add_argument("--test_during_training", action='store_true',
                        help='flag whether to test during training')
    parser.add_argument('--disable_gpu', action='store_true',
                        help='flag whether to disable GPU')
    parser.add_argument('--multi_gpu', action='store_true',
                        help='flag whether to use multiple GPUs')
    # data info
    parser.add_argument('--database', default='PIPAL2DCN+0.5QKV', type=str,
                        help='database name')
    # model info
    parser.add_argument('--model', default='SDSA', type=str,
                        help='model name')  # WResNet  SDSA
    # using less data for testing code  False跑全部数据，True跑小的
    parser.add_argument('--use_less_data', default=False, type=bool,
                        help='using less data for testing code')
    parser.add_argument('--paint_test', default=True, type=bool,
                        help='绘图和测试模式，不进行训练')
    parser.add_argument('--paint_dir', default="./visualize_data/PIPAL", type=str,
                        help='失真图片路径，画图测试不进行训练')
    parser.add_argument('--paint_ref_dir', default="./visualize_data/REF", type=str,
                        help='参考图片路径，画图测试不进行训练')


    args = parser.parse_args()

    args.patch_size = 32  # 32

    args.n_patches = 32
    # args.batch_size = 2
    args.weighted_average = True
    args.paint_lists = [[] for _ in range(5+2)]
    args.ori_imgs = []

    # PIPAL dataset
    if args.database == 'PIPAL2DCN+0.5QKV':
        # args.data_info = './data/PIPAL_TR.mat'
        args.data_info = './data/LIVE.mat'
        args.im_dir = 'D:/1pengchang/shujuji/LIVE/distorted_images'
        # args.ref_dir = './LIVE/distorted_images/refimgs'
        args.ref_dir = None
        #args.im_dir = 'D:/1pengchang/shujuji/PIPAL/PIPAL/Distortion/'
        #args.ref_dir = 'D:/1pengchang/shujuji/PIPAL/PIPAL/Train_Ref/'
    # part of the PIPAL dataset used for validating
    elif args.database == 'PIPAL2':

        args.data_info = './data/PIPAL2.mat'
        args.im_dir = 'D:/0000PZQS/P/shujuji/PIPAL/NTIRE2022_FR_Valid_Dis'
        args.ref_dir = 'D:/0000PZQS/P/shujuji/PIPAL/NTIRE2022_FR_Valid_Ref'
        #args.im_dir = 'D:/1pengchang/shujuji/PIPAL/PIPAL/NTIRE2022_FR_Valid_Dis/'
        #args.ref_dir = 'D:/1pengchang/shujuji/PIPAL/PIPAL/NTIRE2022_FR_Valid_Ref/'
    args.log_dir = '{}/EXP{}-{}-{}-{}-lr={}-bs={}'.format(args.log_dir, args.exp_id, args.k_test, args.database,
                                                          args.model, args.lr, args.batch_size)

    mkdirs('dwcheckpoints')
    args.trained_model_file = 'dwcheckpoints/{}-{}-EXP{}-{}-lr={}-bs={}'.format(args.model, args.database, args.exp_id,
                                                                                args.k_test, args.lr, args.batch_size)
    # logs
    if not os.path.exists('./dwresults'):
        mkdirs('dwresults')  # 创建
    # 如果文件存在则覆盖，否则创建
    filename = '{}-{}-{}-{}.txt'.format(args.model, args.database, args.lr, args.batch_size)
    with open('./dwresults/' + filename, 'w') as fp:
        pass
    f = open('./dwresults/' + filename, 'a+')  # a+是对文件读写
    f.write('{}-{}-{}-{}'.format(args.model, args.database, args.lr, args.batch_size) + '\n')
    f.close()
    args.save_result_file = 'dwresults/{}-{}-EXP{}-{}-lr={}-bs={}'.format(args.model, args.database, args.exp_id,
                                                                          args.k_test, args.lr, args.batch_size)

    # random seed
    args.seed = random.randint(0, 99999999)
    print('Current Random Seed: {}'.format(args.seed))
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

    run(args)



