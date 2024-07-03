'''
This is our method core code for selection and learning,total code will be open after accepted.Thanks
'''
import copy
import torch.nn as nn
from loss import WorstCaseEstimationLoss
from utils import WarmStartGradientReverseLayer, ConfidenceBasedSelfTrainingLoss, linear_rampup2
from Randommix import *

class ConfidenceSoftCELoss(object):
    def __call__(self, outputs, targets, threshold=0.1):
        confidence, confidence_pesudo_label = F.softmax(targets.detach(), dim=1).max(dim=1)
        mask = (confidence > threshold).float()
        Lx = (F.cross_entropy(outputs,confidence_pesudo_label,reduction='none') * mask).mean()
        return Lx

class GCELoss(torch.nn.Module):
    def __init__(self, num_classes, q=0.7, gpu=None):
        super(GCELoss, self).__init__()
        self.device = torch.device('cuda:%s'%gpu) if gpu else torch.device('cpu')
        self.num_classes = num_classes
        self.q = q

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        # label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().cuda()
        gce = (1. - torch.pow(torch.sum(labels * pred, dim=1), self.q)) / self.q
        return gce.mean()

class SemiLoss(object):
    def __call__(self, outputs_x, targets_x):
        #, outputs_u, targets_u, epoch, warm_up
        # probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        # Lu = torch.mean((probs_u - targets_u)**2)
        #, Lu, linear_rampup(epoch,warm_up)
        return Lx
class CE_Soft_Label(nn.Module):
    def __init__(self,num_class = 100):
        super().__init__()
        # print('Calculating uniform targets...')
        # calculate confidence
        self.confidence = None
        self.gamma = 2.0
        self.alpha = 0.25
        self.num_class = num_class
    def init_confidence(self, noisy_labels, num_class):
        noisy_labels = torch.Tensor(noisy_labels).long().cuda()
        self.confidence = F.one_hot(noisy_labels, num_class).float().clone().detach()

    def forward(self, outputs, targets=None):
        # targets = F.one_hot(targets,self.num_class).float()
        logsm_outputs = F.log_softmax(outputs, dim=1)
        final_outputs = logsm_outputs * targets.detach()
        loss_vec = - ((final_outputs).sum(dim=1))
        #p = torch.exp(-loss_vec)
        #loss_vec =  (1 - p) ** self.gamma * loss_vec
        average_loss = loss_vec.mean()
        return loss_vec

    @torch.no_grad()
    def confidence_update(self, temp_un_conf, batch_index, conf_ema_m):
        with torch.no_grad():
            _, prot_pred = temp_un_conf.max(dim=1)
            pseudo_label = F.one_hot(prot_pred, temp_un_conf.shape[1]).float().cuda().detach()
            self.confidence[batch_index, :] = conf_ema_m * self.confidence[batch_index, :]\
                 + (1 - conf_ema_m) * pseudo_label
        return None
def normalize_l2(x, axis=1):
    '''x.shape = (num_samples, feat_dim)'''
    x_norm = np.linalg.norm(x, axis=axis, keepdims=True)
    x = x / (x_norm + 1e-8)
    return x


class dop_model(nn.Module):
    def __init__(self, model, num_classes,args = None, plus=False,mix = False):
        super(dop_model, self).__init__()
        num_examp = 50000
        self.num_classes = num_classes
        self.encoder = model.cuda()
        num_ftrs = self.encoder.linear.in_features
        self.num_ftrs = num_ftrs
        self.cfc = copy.deepcopy(self.encoder.linear).cuda()
        self.encoder.linear = nn.Identity().cuda()
        #add ema to model
        self.encoder_ema = copy.deepcopy(self.encoder).cuda()
        self.cfc_ema = copy.deepcopy(self.cfc).cuda()
        mlp_dim = num_ftrs * 2
        self.pfc = nn.Sequential(
            nn.Linear(num_ftrs, mlp_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(mlp_dim, self.num_classes)
        ).cuda()
        # self.pfc = nn.Linear(num_ftrs,self.num_classes)

        self.worst_head = nn.Sequential(
            nn.Linear(num_ftrs, mlp_dim),
            nn.BatchNorm1d(mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, mlp_dim),
            nn.BatchNorm1d(mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, self.num_classes)
        ).cuda()
        self.grl_layer = WarmStartGradientReverseLayer(alpha=1.0, lo=0.0, hi=args.hi, max_iters=args.warm_iter, auto_step=False)
        self.worst_loss = WorstCaseEstimationLoss(args.eta_prime)
        self.dr = nn.Linear(num_ftrs, args.dr_dim)
        self.u = nn.Parameter(torch.empty(num_examp, 1, dtype=torch.float32))
        self.v = nn.Parameter(torch.empty(num_examp, num_classes, dtype=torch.float32))
        self.m = nn.Parameter(torch.eye(num_classes,num_classes,dtype=torch.float32))
        self.target = torch.zeros(num_examp, self.num_classes).cuda()
        self.beta = 0.9
        self.init_param(mean=0.0, std=1e-8)
        self.args = args
        self.plus = plus
        self.mix = mix
        self.CEsoft = CE_Soft_Label(num_class=num_classes)
        self.args = args
        self.gce_loss = GCELoss(num_classes,0.7,args.gpuid).cuda()
        # dynamic correction for noise labels
        self.w_prev_confidence = torch.ones(num_examp).cuda() * 1 / num_examp
        self.momentum = 0.99
        self.decay = 0.999
        self.sigma = 0.6


    def init_param(self, mean=0., std=1e-8):
        torch.nn.init.normal_(self.u, mean=mean, std=std)
        torch.nn.init.normal_(self.v, mean=mean, std=std)
    def consistency_loss(self, output1, output2):
        preds1 = F.softmax(output1, dim=1).detach()
        preds2 = F.log_softmax(output2, dim=1)
        loss_kldiv = F.kl_div(preds2, preds1, reduction='none')
        loss_kldiv = torch.sum(loss_kldiv, dim=1)
        return loss_kldiv

    def soft_to_hard(self, x):
        with torch.no_grad():
            return (torch.zeros(len(x), self.num_classes)).cuda().scatter_(1, (x.argmax(dim=1)).view(-1, 1), 1)

    @torch.no_grad()
    def momentum_update_ema(self):
        for param_train, param_eval in zip(self.encoder.parameters(), self.encoder_ema.parameters()):
            param_eval.copy_(param_eval * self.decay + param_train.detach() * (1 - self.decay))
        for buffer_train, buffer_eval in zip(self.encoder.buffers(), self.encoder_ema.buffers()):
            buffer_eval.copy_(buffer_eval * self.decay + buffer_train * (1 - self.decay))
        for param_train, param_eval in zip(self.cfc.parameters(), self.cfc_ema.parameters()):
            param_eval.copy_(param_eval * self.decay + param_train.detach() * (1 - self.decay))
        for buffer_train, buffer_eval in zip(self.cfc.buffers(), self.cfc_ema.buffers()):
            buffer_eval.copy_(buffer_eval * self.decay + buffer_train * (1 - self.decay))


    def step(self):
        self.grl_layer.step()
    def forward(self, data, mode='train',epoch = 0):
        if mode == 'train':
            self.momentum_update_ema()
            c_w, c_s, n_w, n_s, c_t, n_t, index,prob_loss= [x.cuda(non_blocking=True) for x in data]
            prob_loss = prob_loss.cuda()
            if len(n_s) > 1 and len(c_w) > 1:
                # n_s_f = self.encoder(n_s)
                n_w_f = self.encoder(n_w)
                n_w_cfc = self.cfc(n_w_f)
                # n_s_pfc = self.pfc(n_s_f)
                n_w_pfc = self.pfc(n_w_f)
                c_w_f = self.encoder(c_w)
                c_w_cfc = self.cfc(c_w_f)
                output = torch.cat((c_w_cfc, n_w_cfc), dim=0)
                target = torch.cat((c_t, n_t))
                c_f_adv = self.grl_layer(c_w_f)
                n_f_adv = self.grl_layer(n_w_f)
                y_adv_c = self.worst_head(c_f_adv)
                y_adv_n = self.worst_head(n_f_adv)

            elif len(c_w) <= 1:
                sample = torch.cat((c_w, n_w), dim=0)
                feature = self.encoder(sample)
                output = self.cfc(feature)
                target = torch.cat((c_t, n_t))

            else:
                sample= torch.cat((c_w, n_w)) if len(n_w) > 0 else c_w
                feature = self.encoder(sample)
                output = self.cfc(feature)
                target = torch.cat((c_t, n_t)) if len(n_t) > 0 else c_t
            eps = 1e-4
            matrix = (self.m - self.m.min()) / (self.m.max() - self.m.min())
            label = torch.zeros(len(target), self.num_classes).cuda().scatter_(1, target.view(-1, 1), 1).cuda()
            U_square = self.u[index] ** 2 * label
            V_square = self.v[index] ** 2 * (1 - label)
            original_prediction = F.softmax(output, dim=1)
            label_one_hot = self.soft_to_hard(output.detach())
            U_square = torch.clamp(U_square, 0, 1)
            V_square = torch.clamp(V_square, 0, 1)

            prediction = torch.clamp(original_prediction + U_square - V_square.detach(), min=eps)
            prediction = F.normalize(prediction, p=1, eps=eps)
            prediction = torch.clamp(prediction, min=eps, max=1.0)
            loss = torch.mean(-torch.sum((label) * torch.log(prediction), dim=-1))
            MSE_loss = F.mse_loss((label_one_hot + U_square - V_square), label, reduction='sum') / len(label)
            loss_mix = 0.
            loss_total_mix = 0.
            loss_for_worst = 0.
            img = torch.cat((c_w, n_w), dim=0)
            c_w_loss = img[prob_loss == 0]
            n_w_loss = img[prob_loss == 1]
            c_t_loss = target[prob_loss == 0]
            c_w_cfc_loss = output[prob_loss == 0]
            n_w_cfc_loss = output[prob_loss == 1]
            if epoch >= self.args.warm_u + self.args.warm_up and len(c_w_loss) > 1 and len(n_w_loss) > 1:
                w = linear_rampup2(epoch, self.args.mix, start=self.args.warm_u + self.args.warm_up)
                # label refinement of clean samples
                with torch.no_grad():
                    # clean choose
                    y_true = F.one_hot(c_t_loss, self.num_classes).float()
                    px = torch.softmax(c_w_cfc_loss, dim=1)
                    loss_1 = -torch.sum((y_true) * torch.log(px), dim=-1)
                    ind_1_sorted = np.argsort(loss_1.data).cuda()
                    clean_num = int(0.9 * len(loss_1))
                    idx_chosen_c = ind_1_sorted[:clean_num]
                    # -----------------dynamic threshold-----------------------------#
                    index_n = index[c_w.shape[0]:]
                    confidence, pesudo_label_n = F.softmax(n_w_cfc_loss.detach(), dim=1).max(dim=1)
                    ws_threshold = self.w_prev_confidence[index_n] + 0.4
                    dynamic_threshold = torch.min(ws_threshold, torch.tensor(self.args.tau)).cuda()
                    idx_chosen_n = confidence > dynamic_threshold
                    pesudo_label_n = F.one_hot(pesudo_label_n, self.num_classes).float()
                l = np.random.beta(4, 4)
                l = max(l, 1 - l)
                X_w_c = c_w_loss[idx_chosen_c]
                pesudo_label_c = y_true[idx_chosen_c]
                X_w_n = n_w_loss[idx_chosen_n]
                pesudo_label_n = pesudo_label_n[idx_chosen_n]
                X_w_c = torch.cat((X_w_c,X_w_n),dim = 0)
                pesudo_label_c = torch.cat((pesudo_label_c,pesudo_label_n),dim = 0)
                if X_w_c.shape[0] > 1:
                    idx = torch.randperm(X_w_c.size(0))
                    X_w_c_rand = X_w_c[idx]
                    # pesudo_label_c = pesudo_label_c[idx]
                    (mix_img1,mix_label1,_,_,lam1),(mix_img2,mix_label2,_,_,lam2) = randommix(X_w_c,pesudo_label_c)
                    mix1_output = self.cfc(self.encoder(mix_img1))
                    mix2_output = self.cfc(self.encoder(mix_img2))
                    loss_mix = self.CEsoft(mix1_output,mix_label1).mean()
                    loss_fmix = self.CEsoft(mix2_output, mix_label2).mean()
                    loss_ce = self.CEsoft(c_w_cfc_loss[idx_chosen_c], targets=y_true[idx_chosen_c]).mean()
                    loss_total_mix = w * (loss_ce + loss_mix + loss_fmix)
            # loss for dst :
            if epoch >= self.args.warm_u + self.args.warm_up and len(c_w_loss) > 1 and len(n_w_loss) > 1:
                _,predict = torch.max(c_w_cfc,dim = -1)
                y_true = F.one_hot(predict, self.num_classes).float()
                y_noise = torch.softmax(n_w_cfc,dim = -1)

                loss_for_worst = self.worst_loss(y_true, y_adv_c, y_noise, y_adv_n,mix = False)

            avg_prediction = torch.mean(prediction, dim=0)
            prior_distr = 1.0/self.num_classes * torch.ones_like(avg_prediction)
            avg_prediction = torch.clamp(avg_prediction, min = eps, max = 1.0)
            balance_reg = torch.mean(-(prior_distr * torch.log(avg_prediction)).sum(dim=0))
            img_s = torch.cat((c_s, n_s), dim=0)
            img_s_feature = self.encoder(img_s)
            output1 = self.cfc(img_s_feature)
            consistency_loss = self.consistency_loss(output, output1).mean()
            # consistency_loss += self.confidence_ce(output_pfc, output,threshold=self.args.tau)
            loss_for_kl = self.args.c * consistency_loss + self.args.reg * balance_reg
            total_loss = loss + MSE_loss + self.args.worst_weight *loss_for_worst + loss_total_mix + loss_for_kl

            with torch.no_grad():
                w_prob_max,_ = torch.max(output.detach(),dim = 1)
                self.w_prev_confidence[index] = self.momentum * self.w_prev_confidence[index] + (1 - self.momentum) * w_prob_max
            loss_dict = {
                "cleanoutput": output,
                "total_loss": total_loss,
                "pce": loss,
                "pmse": MSE_loss,
                "loss_worst": 0.3 * loss_for_worst,
                "target": target,
                "loss_mix":loss_mix,
                "loss_kl":loss_for_kl,
                "loss_pesudo":0,
                "matrix":0
            }
            return loss_dict
        elif mode == 'test' or mode == 'warmup':
            img = data.cuda()
            f = self.encoder_ema(img)
            output = self.cfc_ema(f)
            dr_feature = self.dr(f)
            resdict = {
                'output': output,
                'feature': f,
                "feature_dim": f.shape[-1],
                "dr_feature": dr_feature
            }
            return resdict
        elif mode == 'eval':
            img = data.cuda()
            # f = self.encoder_ema(img)
            # output = self.cfc_ema(f)
            f = self.encoder(img)
            output = self.cfc(f)
            dr_feature = self.dr(f)
            resdict = {
                'output': output,
                'feature': f,
                "feature_dim": f.shape[-1],
                "dr_feature": dr_feature
            }
            return resdict
        else:
            raise "value error,should be in [train,test,warmup]"
