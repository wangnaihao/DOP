from __future__ import print_function
import torch.optim as optim
import argparse
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
import time
from PreResNet import *
from sklearn.mixture import GaussianMixture
from DNLL import dop_model
import dataloader_cifar_all as dataloader_all
import wandb
import copy
parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')


#new args
parser.add_argument('--strong_type', default='rand', type=str, help='rand,augto')
parser.add_argument('--cifarn_mode',type=str,help='clean,arrge,worst,rand1,rand2,rand3,clean100,noise100',default='clean')
parser.add_argument('--select_func', default='u', type=str, help='loss,u,vote,twoview')
parser.add_argument('--batch_size', default=128, type=int, help='train batchsize')
parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, help='initial learning rate')
parser.add_argument('--noise_mode', default='sym', type=str,help='sym,asym,cifar10n,cifar100n,idn')
parser.add_argument('--save_name', type=str, default='test')
parser.add_argument('--num_epochs', default=500, type=int)
# noise rate
parser.add_argument('--r', default=0.5, type=float, help='noise ratio')
parser.add_argument('--id', default='')
parser.add_argument('--seed', default=123, type=int)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_class', default=100, type=int)
parser.add_argument('--data_path', default='data/cifar100', type=str, help='path to dataset')
parser.add_argument('--dataset', default='cifar100', type=str)
parser.add_argument('--warm_u', default=0, type=int)
parser.add_argument('--lr_u', default=1, type=int)
parser.add_argument('--lr_v', default=10, type=int)
parser.add_argument('--lr_m', default=0.001, type=float)
parser.add_argument('--dr_dim', default=128, type=int)
parser.add_argument('--train_type', default="sop", type=str)
parser.add_argument('--eta_prime', default=2, type=float)
parser.add_argument('--warm_iter', default=100000, type=int)
parser.add_argument('--warm_up', default=0, type=int)
parser.add_argument('--opti_type', default="cos", type=str, help="cos or multistep")
parser.add_argument('--mix', default=50, type=float, help="mix weight")
parser.add_argument('--hi', default=0.1, type=float, help="mix weight")
parser.add_argument('--worst_weight', default=0.3, type=float, help="worst weight")
parser.add_argument('--pesudo_weight', default=0.3, type=float, help="worst weight")
parser.add_argument('--tau', default=0.9, type=float, help="select confience label")
parser.add_argument('--c', default=0.9, type=float, help="kl")
parser.add_argument('--reg', default=0.1, type=float, help="plus reg")
parser.add_argument('--stopm', default=60000, type=int, help="stop matrix")
parser.add_argument('--ssl', default=1, type=int, help="stop matrix")
args = parser.parse_args()
if args.ssl == 1:
    ssl = True
else:
    ssl = False
wandb = wandb.init(
    project='co-train-resnet18' + args.dataset + args.noise_mode,
    config=args,
    name=args.train_type + "_" + args.noise_mode + "_" + str(args.r),
    # resume = True,
)
# torch.cuda.set_device(args.gpuid)
# random.seed(args.seed)
# np.random.seed(args.seed)
# os.environ['PYTHONHASHSEED'] = str(args.seed)
# torch.manual_seed(args.seed)
# torch.cuda.manual_seed(args.seed)
# # torch.cuda.manual_seed_all(args.seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# # torch.backends.cudnn.enabled = True
# os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
# torch.use_deterministic_algorithms(True)


def top1acc(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct, len(target)


def top5acc(output, target, k=5):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct, target


class prettyfloat(float):
    # 用来指定float显示的位数
    def __repr__(self):
        return "%0.2f" % self


def two_view_select(model,u,eval_loader,all_loss):
    model.eval()
    pre = np.ones(50000)
    pre_loss = np.ones(50000)
    input_u = torch.abs(u.detach()).cpu().numpy().squeeze(1)
    input_u = input_u.reshape(-1, 1)

    losses = torch.zeros(50000)
    with torch.no_grad():
        for batch_idx, (inputs, targets, index,_) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            resdict = model(inputs, mode='eval')
            outputs = resdict['output']
            loss = CE(outputs, targets)
            for b in range(inputs.size(0)):
                losses[index[b]] = loss[b]

    losses = (losses - losses.min()) / (losses.max() - losses.min())
    all_loss.append(losses)

    if args.r == 0.9:  # average loss over last 5 epochs to improve convergence stability
        history = torch.stack(all_loss)
        input_loss = history[-5:].mean(0)
        input_loss = input_loss.reshape(-1, 1)
    else:
        input_loss = losses.reshape(-1, 1)

    gmm_u = GaussianMixture(n_components=2, max_iter=20, tol=1e-2, reg_covar=5e-4)
    gmm_u.fit(input_u)
    prob_u = gmm_u.predict_proba(input_u)
    prob_u = prob_u[:, gmm_u.means_.argmin()]

    gmm_loss = GaussianMixture(n_components=2, max_iter=20, tol=1e-2, reg_covar=5e-4)
    gmm_loss.fit(input_loss)
    prob_loss = gmm_loss.predict_proba(input_loss)
    prob_loss = prob_loss[:, gmm_loss.means_.argmin()]
    pre[prob_u >= 0.5] = 0
    print(f"u num clean:{np.sum(pre == 0)}")
    print(f"u num noise:{np.sum(pre == 1)}")
    pre_loss[prob_loss >= 0.5] = 0
    print(f"loss num clean:{np.sum(pre_loss == 0)}")
    print(f"loss num noise:{np.sum(pre_loss == 1)}")
    return pre,prob_u,pre_loss,all_loss_1


class SoftCELoss(object):
    def __call__(self, outputs, targets):
        probs = torch.softmax(outputs, dim=1)
        Lx = -torch.mean(torch.sum(F.log_softmax(outputs, dim=1) * targets, dim=1))
        return Lx

def pandr(ct,nt,gt_c,gt_n):
    tp = torch.sum(ct == gt_c)
    fp = torch.sum(ct != gt_c)
    tn = torch.sum(nt != gt_n)
    fn = torch.sum(nt == gt_n)
    percision = tp / (tp + fp)
    recall = tp / (tp + fn)
    acc = (tp + tn) / (tp+fp+tn+fn)
    return percision,recall,acc


def train(epoch, model, optimizer_f, optimizer_cfc, optimizer_pfc, optimizer_uv, optimizer_worst,labeled_trainloader,prob=[]):
    model.train()
    total = 0
    correct = 0
    noise_cnt = 0
    total_sample = 0
    percision = 0.
    recall = 0.
    true_acc = 0.
    batch_id = 0
    labeled_train_iter = iter(labeled_trainloader)
    I = 1
    noise_idx = []

    while True:
        try:
            # 输入图像,标签,权重
            inputs, labels_x, pre, index, gt = labeled_train_iter.next()
            inputs_w, inputs_s = inputs
        except:
            if I == 1:
                acc = 100. * correct / total
                print("------------------------------------")
                print(f"train epoch : {epoch}")
                print(f"true noise_rate : {noise_cnt / total_sample}")
                print(f"total_loss : {resdict['total_loss']}")
                print(f"pce : {resdict['pce']}")
                print(f"pmse : {resdict['pmse']}")
                print(f"loss_worst : {resdict['loss_worst']}")
                print(f"loss_mix : {resdict['loss_mix']}")
                print(f"loss_kl : {resdict['loss_kl']}")
                print(f"loss_pesudo : {resdict['loss_pesudo']}")
                print(f"train acc : {acc}")
                print(f"matrix:{resdict['matrix']}")
                # log = {
                #     "train_epoch": epoch,
                #     "train_acc": acc,
                #     "total_loss": resdict['total_loss'],
                #     "percision": percision / batch_id,
                #     "recall": recall / batch_id,
                #     "True acc": true_acc / batch_id
                # }
                # wandb.log(log)
                # if epoch >= args.warm_u + args.warm_up:
                #     print(f"correct label to gt : {round(100.* label_correct_right / correct_label_cnt,2)}%")
                #     log.update({"correct2right_percent" :round(100.* label_correct_right / correct_label_cnt,2) })
                print("------------------------------------")
                print()
                # stats_log.write(f"------------------------------------\n")
                # for key in log.keys():
                #     stats_log.write(f"{str(key)} : {str(log[key])}" + "\n")

                # stats_log.flush()
                return
            else:
                labeled_train_iter = iter(labeled_trainloader)
                inputs, labels_x, pre, index, gt = labeled_train_iter.next()
                # inputs --> (img_w,img_s)
                inputs_w, inputs_s = inputs
                I = I - 1
        gt = gt
        gt_c = gt[pre == 0]
        gt_n = gt[pre == 1]
        prob1 = prob[index]
        c_w = inputs_w[pre == 0]
        c_s = inputs_s[pre == 0]
        n_w = inputs_w[pre == 1]
        n_s = inputs_s[pre == 1]
        c_label = labels_x[pre == 0]
        n_label = labels_x[pre == 1]
        index_c = index[pre == 0]
        index_n = index[pre == 1]
        c_prob = prob1[pre == 0]
        n_prob = prob1[pre == 1]
        prob_loss = torch.tensor(np.concatenate((c_prob, n_prob), axis=0))
        noise_idx.extend(index_n.tolist())
        _index = torch.cat((index_c, index_n)) if index_n.size(0) > 0 else index_c
        label = torch.cat((c_label, n_label))
        labels_x = labels_x
        p,r,a = pandr(c_label,n_label,gt_c,gt_n)
        percision += p
        recall += r
        true_acc += a
        batch_id += 1

        data = [c_w, c_s, n_w, n_s, c_label, n_label, _index, prob_loss]
        noise_cnt += torch.sum(labels_x != gt).item()
        total_sample += len(gt)
        resdict = model(data,mode='train', epoch=epoch)
        loss = resdict['total_loss']
        # print(loss)
        output = resdict['cleanoutput']
        target = resdict['target']
        if epoch >= args.warm_u + args.warm_up:
            optimizer_f.zero_grad()
            optimizer_cfc.zero_grad()
            optimizer_pfc.zero_grad()
            optimizer_uv.zero_grad()
            optimizer_worst.zero_grad()
            loss.backward()
            optimizer_f.step()
            optimizer_cfc.step()
            optimizer_pfc.step()
            optimizer_uv.step()
            optimizer_worst.step()
            # update factor of gradient reverse layer
            model.step()
        else:
            optimizer_f.zero_grad()
            optimizer_cfc.zero_grad()
            optimizer_uv.zero_grad()
            loss.backward()
            optimizer_f.step()
            optimizer_cfc.step()
            optimizer_uv.step()
        right, cnt = top1acc(output, label.cuda())
        correct += right
        total += cnt


CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()
softCE = SoftCELoss()
@torch.no_grad()
def test(epoch, net1,net2, best_acc):
    net1.eval()
    net2.eval()
    correct1 = 0
    correct2 = 0
    correct_ave = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(test_loader):
            targets, index = targets.cuda(), index.cuda()
            resdict1 = net1(inputs, mode='test')
            resdict2= net2(inputs, mode='test')
            outputs1 = resdict1['output']
            _, predicted1 = torch.max(outputs1, 1)
            outputs2 = resdict2['output']
            _, predicted2 = torch.max(outputs2, 1)
            ave_output = (outputs1 + outputs2) / 2
            _, predicted_ave = torch.max(ave_output, 1)
            total += targets.size(0)
            correct1 += predicted1.eq(targets).cpu().sum().item()
            correct2 += predicted2.eq(targets).cpu().sum().item()
            correct_ave += predicted_ave.eq(targets).cpu().sum().item()
    acc1 = 100. * correct1 / total
    acc2 = 100. * correct2 / total
    acc_ave = 100. * correct_ave / total
    wandb.log({
        "test_acc1": acc1,
        "test_acc2":acc2,
        "test_acc_ave":acc_ave
    })
    if acc_ave > best_acc:
        best_acc = acc_ave
    print("------------------------------------")
    print(f"test epoch : {epoch}")
    print(f"test acc1 : {acc1}")
    print(f"test acc2 : {acc2}")
    print(f"test acc_ave : {acc_ave}")
    print(f"best_acc : {best_acc}")
    print("------------------------------------")
    print()
    return acc1,acc2,acc_ave, best_acc


class DualNet:
    def __init__(self):
        resnet1 = PreResNet18(num_classes=args.num_class)
        self.model1 = dop_model(resnet1, args.num_class, args, plus=ssl, mix=args.mix).cuda()
        resnet2 = PreResNet18(num_classes=args.num_class)
        self.model2 = dop_model(resnet2, args.num_class, args, plus=ssl, mix=args.mix).cuda()
    def forward(self,x):
        output1 = self.model1(x, mode = 'test')['output']
        output2 = self.model2(x, mode = 'test')['output']
        output_mean = (output1 + output2) / 2
        return output_mean

start_time = time.strftime('%Y-%m-%d %H:%M:%S')
stats_log = open('./checkpoint/%s_%s_%.1f_%s_%s' % (
args.save_name, args.dataset, args.r, args.noise_mode, start_time) + '_stats.txt',
                 'w')


loader = dataloader_all.cifar_dataloader(args.dataset, r=args.r, noise_type=args.noise_mode,cifarn_mode=args.cifarn_mode,\
                                     batch_size=args.batch_size,strong_type = args.strong_type, \
                                     num_workers=6, root_dir=args.data_path, \
                                     noise_file='%s/%.1f_%s.json' % (args.data_path, args.r, args.noise_mode))


print('| Building net')

dual_net = DualNet()

test_loader = loader.run('test')

optimizer_f1= optim.SGD(list(dual_net.model1.encoder.parameters()), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer_cfc1 = optim.SGD(list(dual_net.model1.cfc.parameters()), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer_pfc1 = optim.SGD(list(dual_net.model1.pfc.parameters()), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer_worst1 = optim.SGD(list(dual_net.model1.worst_head.parameters()), lr=0.03, momentum=0.9, weight_decay=5e-4)

optimizer_f2 = optim.SGD(list(dual_net.model2.encoder.parameters()), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer_cfc2 = optim.SGD(list(dual_net.model2.cfc.parameters()), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer_pfc2 = optim.SGD(list(dual_net.model2.pfc.parameters()), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer_worst2 = optim.SGD(list(dual_net.model2.worst_head.parameters()), lr=0.03, momentum=0.9, weight_decay=5e-4)
if args.opti_type == "cos":
    args.nums_epochs = 300
    lr_scheduler_f1 = CosineAnnealingLR(optimizer=optimizer_f1, T_max=300, eta_min=0.0002)

    lr_scheduler_cfc1 = CosineAnnealingLR(optimizer=optimizer_cfc1, T_max=300, eta_min=0.0002)
    lr_scheduler_pfc1 = CosineAnnealingLR(optimizer=optimizer_pfc1, T_max=300, eta_min=0.0002)
    lr_scheduler_worst1 = CosineAnnealingLR(optimizer=optimizer_worst1, T_max=300, eta_min=0.0002)

    lr_scheduler_f2= CosineAnnealingLR(optimizer=optimizer_f2, T_max=300, eta_min=0.0002)

    lr_scheduler_cfc2 = CosineAnnealingLR(optimizer=optimizer_cfc2, T_max=300, eta_min=0.0002)
    lr_scheduler_pfc2 = CosineAnnealingLR(optimizer=optimizer_pfc2, T_max=300, eta_min=0.0002)
    lr_scheduler_worst2 = CosineAnnealingLR(optimizer=optimizer_worst2, T_max=300, eta_min=0.0002)

elif args.opti_type == "multi_step":
    args.nums_epochs = 150
    lr_scheduler_f1 = MultiStepLR(optimizer=optimizer_f1, milestones=[40, 80], gamma=0.1)

    lr_scheduler_cfc1 = MultiStepLR(optimizer=optimizer_cfc1, milestones=[40, 80], gamma=0.1)
    lr_scheduler_pfc1 = MultiStepLR(optimizer=optimizer_pfc1, milestones=[40, 80], gamma=0.1)
    lr_scheduler_worst1 = MultiStepLR(optimizer=optimizer_worst1, milestones=[40, 80], gamma=0.1)

    lr_scheduler_f2 = MultiStepLR(optimizer=optimizer_f2, milestones=[40, 80], gamma=0.1)

    lr_scheduler_cfc2 = MultiStepLR(optimizer=optimizer_cfc2, milestones=[40, 80], gamma=0.1)
    lr_scheduler_pfc2 = MultiStepLR(optimizer=optimizer_pfc2, milestones=[40, 80], gamma=0.1)
    lr_scheduler_worst2 = MultiStepLR(optimizer=optimizer_worst2, milestones=[40, 80], gamma=0.1)
else:
    raise "value should be in [multi_step,cos]"
reparam_params1 = [{'params': dual_net.model1.u, 'lr': args.lr_u, 'weight_decay': 0},
                  {'params': dual_net.model1.v, 'lr': args.lr_v, 'weight_decay': 0},
                  {'params': dual_net.model1.m, 'lr': args.lr_m, 'weight_decay': 0}
                  ]  # , 'momentum': config['optimizer_overparametrization']['args']['momentum']}]

optimizer_uv1 = optim.SGD(reparam_params1, lr=1, momentum=0, weight_decay=0)

reparam_params2 = [{'params': dual_net.model2.u, 'lr': args.lr_u, 'weight_decay': 0},
                  {'params': dual_net.model2.v, 'lr': args.lr_v, 'weight_decay': 0},
                  {'params': dual_net.model2.m, 'lr': args.lr_m, 'weight_decay': 0}
                  ]  # , 'momentum': config['optimizer_overparametrization']['args']['momentum']}]

optimizer_uv2 = optim.SGD(reparam_params2, lr=1, momentum=0, weight_decay=0)
best_acc = 0.
all_loss_1 = []
all_loss_2 = []

for epoch in range(0, args.num_epochs):
    eval_loader = loader.run('eval_train')
    start_time = time.time()
    print('Train Epoch: {}'.format(epoch))

    print('Train WangnhMdoel')
    if epoch >= args.warm_u + args.warm_up:
        print("epoch for divide dataset to clean and noise by twoview")
        u1 = dual_net.model1.u.detach()
        u2 = dual_net.model2.u.detach()
        pre1,prob1,pre_loss1,all_loss_1 = two_view_select(dual_net.model1,u1,eval_loader,all_loss_1)
        pre2,prob2,pre_loss2,all_loss_2 = two_view_select(dual_net.model2,u2,eval_loader,all_loss_2)
        labeled_trainloader1 = loader.run('train', pre1)
        labeled_trainloader2 = loader.run('train', pre2)
    else:
        print("epoch for warm u")
        u = dual_net.model1.u.detach()
        pre1 = np.zeros(u.shape[0])
        prob1 = np.zeros(u.shape[0])
        pre2 = np.zeros(u.shape[0])
        prob2 = np.zeros(u.shape[0])
        pre_loss1 = np.zeros(u.shape[0])
        pre_loss2 = np.zeros(u.shape[0])
        labeled_trainloader1 = loader.run('train', pre1)
        labeled_trainloader2 = copy.deepcopy(labeled_trainloader1)


    print("train net2...")
    train(epoch, dual_net.model2, optimizer_f2, optimizer_cfc2, optimizer_pfc2, optimizer_uv2, optimizer_worst2,
          labeled_trainloader1, prob=pre_loss2)

    print("train net1...")
    train(epoch, dual_net.model1, optimizer_f1, optimizer_cfc1, optimizer_pfc1, optimizer_uv1, optimizer_worst1,
          labeled_trainloader2,prob=pre_loss1)

    lr_scheduler_f1.step()
    lr_scheduler_cfc1.step()
    lr_scheduler_f2.step()
    lr_scheduler_cfc2.step()
    if epoch >= args.warm_u + args.warm_up:
        lr_scheduler_pfc1.step()
        lr_scheduler_worst1.step()
        lr_scheduler_pfc2.step()
        lr_scheduler_worst2.step()
    print(f"now lr is : {lr_scheduler_f1.get_last_lr()}")
    end_time = time.time()
    run_time = end_time - start_time  # 程序的运行时间，单位为秒
    m = int(run_time // 60)
    s = int(run_time % 60)
    print(f"this epoch run time : {m}m{s}s")
    acc1,acc2,acc_ave, best_acc = test(epoch, dual_net.model1,dual_net.model2, best_acc)