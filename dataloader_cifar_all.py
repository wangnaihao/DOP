from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import os
import torch
from numpy.testing import assert_array_almost_equal
import copy
from randaugment import RandAugment


noise_type_map = {'clean': 'clean_label', 'worst': 'worse_label', 'aggre': 'aggre_label', 'rand1': 'random_label1',
                  'rand2': 'random_label2', 'rand3': 'random_label3', 'clean100': 'clean_label',
                  'noisy100': 'noisy_label'}


def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict


class cifar_dataset(Dataset):
    def __init__(self, dataset, r, noise_type, cifarn_mode, root_dir, transform, mode, noise_file='',probability = []):

        self.r = r  # noise ratio
        self.transform = transform
        self.mode = mode
        self.transition = {0: 0, 2: 0, 4: 7, 7: 7, 1: 1, 9: 1, 3: 5, 5: 3, 6: 6,
                           8: 8}  # class transition for asymmetric noise
        self.cifar100_transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
        ])
        self.cifar10_transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ])
        self.probability = probability

        if self.mode == 'test':
            if dataset == 'cifar10':
                test_dic = unpickle('%s/test_batch' % root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))
                self.test_label = test_dic['labels']
            elif dataset == 'cifar100':
                test_dic = unpickle('%s/test' % root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))
                self.test_label = test_dic['fine_labels']
        else:
            train_data = []
            train_label = []
            if dataset == 'cifar10':
                for n in range(1, 6):
                    dpath = '%s/data_batch_%d' % (root_dir, n)
                    data_dic = unpickle(dpath)
                    train_data.append(data_dic['data'])
                    train_label = train_label + data_dic['labels']
                train_data = np.concatenate(train_data)
            elif dataset == 'cifar100':
                train_dic = unpickle('%s/train' % root_dir)
                train_data = train_dic['data']
                train_label = train_dic['fine_labels']
            train_data = train_data.reshape((50000, 3, 32, 32))
            train_data = train_data.transpose((0, 2, 3, 1))

            self.train_data = train_data
            self.clean_label = np.array(train_label)
            if noise_type == 'idn':
                if os.path.exists(noise_file):
                    print("use already exist noise file")
                    self.noise_label = json.load(open(noise_file, "r"))
                else:
                    print("generate random noise")
                    num_classes = 100 if dataset == 'cifar100' else 10
                    self.noise_label = self.instance_noise(tau=self.r, num_classes=num_classes)
                    print("save noisy labels to %s ..." % noise_file)
                    json.dump(self.noise_label, open(noise_file, "w"))
            elif noise_type == 'cifar10n' or noise_type == 'cifar100n':
                self.noise_path = "./data/cifar100N/CIFAR-100_human.pt" if noise_type == 'cifar100n' else "./data/cifar10N/CIFAR-10_human.pt"
                # Load human noisy labels
                noise = torch.load(self.noise_path)
                self.noise_label = noise[noise_type_map[cifarn_mode]].reshape(-1)
            elif noise_type == 'asym' or noise_type == 'sym':
                if os.path.exists(noise_file):
                    print("use already exist noise file")
                    self.noise_label = json.load(open(noise_file, "r"))
                elif dataset == 'cifar100' and noise_type == 'asym':
                    print("generate random noise")
                    self.noise_label = self.asymmetric_noise(np.array(train_label)).tolist()
                    print("save noisy labels to %s ..." % noise_file)
                    json.dump(self.noise_label, open(noise_file, "w"))
                else:  # inject noise
                    print("generate random noise")
                    noise_label = []
                    idx = list(range(50000))
                    random.shuffle(idx)
                    num_noise = int(self.r * 50000)
                    noise_idx = idx[:num_noise]
                    for i in range(50000):
                        if i in noise_idx:
                            if noise_type == 'sym':
                                if dataset == 'cifar10':
                                    noiselabel = random.randint(0, 9)
                                elif dataset == 'cifar100':
                                    noiselabel = random.randint(0, 99)
                                noise_label.append(noiselabel)
                            elif noise_type == 'asym':
                                noiselabel = self.transition[train_label[i]]
                                noise_label.append(noiselabel)
                        else:
                            noise_label.append(train_label[i])
                    self.noise_label = noise_label
                    print("save noisy labels to %s ..." % noise_file)
                    json.dump(self.noise_label, open(noise_file, "w"))


    def __getitem__(self, index):
        if self.mode == 'normal':
            img, target, clean = self.train_data[index], self.noise_label[index], self.clean_label[index]
            img = Image.fromarray(img)
            weak = self.transform(img)
            return weak, target, index, clean
        elif self.mode == 'strong_aug':
            img, target, clean, = self.train_data[index], self.noise_label[index], self.clean_label[index]
            prob = self.probability[index]
            img = Image.fromarray(img)
            weak, strong = self.transform[0](img), self.transform[1](img)
            return (weak, strong), target, prob,index, clean
        elif self.mode == 'test':
            img, target = self.test_data[index], self.test_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)
            return img, target,index
        elif self.mode == 'eval_train':
            img, target, clean = self.train_data[index], self.noise_label[index], self.clean_label[index]
            img = Image.fromarray(img)
            weak = self.transform(img)
            weak_flip = transforms.RandomHorizontalFlip(p=1)(weak)
            return weak, weak_flip,target, index, clean


    def __len__(self):
        if self.mode != 'test':
            return len(self.train_data)
        else:
            return len(self.test_data)

    def multiclass_noisify(self, y, P, random_state=0):
        """ Flip classes according to transition probability matrix T.
        It expects a number between 0 and the number of classes - 1.
        """

        assert P.shape[0] == P.shape[1]
        assert np.max(y) < P.shape[0]

        # row stochastic matrix
        assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
        assert (P >= 0.0).all()

        m = y.shape[0]
        new_y = y.copy()
        flipper = np.random.RandomState(random_state)

        for idx in np.arange(m):
            i = y[idx]
            # draw a vector with only an 1
            flipped = flipper.multinomial(1, P[i, :], 1)[0]
            new_y[idx] = np.where(flipped == 1)[0]

        return new_y

    def build_for_cifar100(self, size, noise):
        """ random flip between two random classes.
        """
        assert (noise >= 0.) and (noise <= 1.)

        P = np.eye(size)
        cls1, cls2 = np.random.choice(range(size), size=2, replace=False)
        P[cls1, cls2] = noise
        P[cls2, cls1] = noise
        P[cls1, cls1] = 1.0 - noise
        P[cls2, cls2] = 1.0 - noise

        assert_array_almost_equal(P.sum(axis=1), 1, 1)
        return P

    def asymmetric_noise(self, train_label, num_classes=100):
        P = np.eye(num_classes)
        n = self.r
        nb_superclasses = 20
        nb_subclasses = 5

        if n > 0.0:
            for i in np.arange(nb_superclasses):
                init, end = i * nb_subclasses, (i + 1) * nb_subclasses
                P[init:end, init:end] = self.build_for_cifar100(nb_subclasses, n)

            y_train_noisy = self.multiclass_noisify(train_label, P=P,
                                                    random_state=0)
            actual_noise = (y_train_noisy != train_label).mean()
            assert actual_noise > 0.0
            return y_train_noisy

    def instance_noise(
            self,
            tau: float = 0.2,
            std: float = 0.1,
            feature_size: int = 3 * 32 * 32,
            # seed: int = 1
            num_samples=50000,
            num_classes=10,
    ):
        '''
        Thanks the code from https://github.com/SML-Group/Label-Noise-Learning wrote by SML-Group.
        LabNoise referred much about the generation of instance-dependent label noise from this repo.
        '''
        from scipy import stats
        from math import inf
        import torch.nn.functional as F

        if num_classes == 10:
            transforms_idn = self.cifar10_transform_test
        elif num_classes == 100:
            transforms_idn = self.cifar100_transform_test
        # common-used parameters
        min_target, max_target = min(self.clean_label), max(self.clean_label)
        P = []
        # sample instance flip rates q from the truncated normal distribution N(\tau, {0.1}^2, [0, 1])
        flip_distribution = stats.truncnorm((0 - tau) / std, (1 - tau) / std,
                                            loc=tau,
                                            scale=std)
        '''
        The standard form of this distribution is a standard normal truncated to the range [a, b]
        notice that a and b are defined over the domain of the standard normal. 
        To convert clip values for a specific mean and standard deviation, use:

        a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std
        truncnorm takes  and  as shape parameters.

        so the above `flip_distribution' give a truncated standard normal distribution with mean = `tau`,
        range = [0, 1], std = `std`
        '''
        # import ipdb; ipdb.set_trace()
        # how many random variates you need to get
        q = flip_distribution.rvs(num_samples)
        # sample W \in \mathcal{R}^{S \times K} from the standard normal distribution N(0, 1^2)
        W = torch.tensor(
            np.random.randn(num_classes, feature_size,
                            num_classes)).float().cuda()  # K*dim*K, dim=3072
        for i in range(num_samples):
            x, y = transforms_idn(Image.fromarray(self.train_data[i])), torch.tensor(self.clean_label[i])
            x = x.cuda()
            # step (4). generate instance-dependent flip rates
            # 1 x feature_size  *  feature_size x 10 = 1 x 10, p is a 1 x 10 vector
            p = x.reshape(1, -1).mm(W[y]).squeeze(0)  # classes
            # step (5). control the diagonal entry of the instance-dependent transition matrix
            # As exp^{-inf} = 0, p_{y} will be 0 after softmax function.
            p[y] = -inf
            # step (6). make the sum of the off-diagonal entries of the y_i-th row to be q_i
            p = q[i] * F.softmax(p, dim=0)
            p[y] += 1 - q[i]
            P.append(p)
        P = torch.stack(P, 0).cpu().numpy()
        l = [i for i in range(min_target, max_target + 1)]
        new_label = [np.random.choice(l, p=P[i]).tolist() for i in range(num_samples)]

        print('noise rate = ', (new_label != np.array(self.clean_label)).mean())
        return new_label


class cifar_dataloader():
    def __init__(self, dataset, r, noise_type, cifarn_mode, batch_size, strong_type, num_workers, root_dir,
                 noise_file=''):
        self.dataset = dataset
        self.r = r
        self.noise_type = noise_type
        self.cifarn_mode = cifarn_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.noise_file = noise_file
        self.cifar10_mean, self.cifar10_std = ((0.4914, 0.4822, 0.4465),(0.247, 0.243, 0.261))
        self.cifar100_mean, self.cifar100_std = (0.507, 0.487, 0.441), (0.267, 0.256, 0.276)
        if self.dataset == 'cifar10':
            self.weak = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),(0.247, 0.243, 0.261)),
            ])

            self.strong = copy.deepcopy(self.weak)
            self.strong.transforms.insert(0, RandAugment(3,5))

            self.transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ])


        elif self.dataset == 'cifar100':
                self.weak = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ])

                self.transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ])
                self.strong = copy.deepcopy(self.weak)
                self.strong.transforms.insert(0, RandAugment(3, 5))

    def run(self, mode,pre = []):
        if mode == 'warmup':
            all_dataset = cifar_dataset(dataset=self.dataset, noise_type=self.noise_type, cifarn_mode=self.cifarn_mode,
                                        r=self.r,
                                        root_dir=self.root_dir, transform=self.weak,
                                        mode="normal", noise_file=self.noise_file)
            trainloader = DataLoader(
                dataset=all_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)
            return trainloader

        elif mode == 'train':
            all_dataset = cifar_dataset(dataset=self.dataset, noise_type=self.noise_type, cifarn_mode=self.cifarn_mode,
                                        r=self.r,
                                        root_dir=self.root_dir, transform=[self.weak, self.strong],
                                        mode="strong_aug", noise_file=self.noise_file,probability=pre)
            trainloader = DataLoader(
                dataset=all_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=self.num_workers)
            return trainloader

        elif mode == 'test':
            test_dataset = cifar_dataset(dataset=self.dataset, noise_type=self.noise_type, cifarn_mode=self.cifarn_mode,
                                         r=self.r,
                                         root_dir=self.root_dir, transform=self.transform_test, mode='test')
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)
            return test_loader

        elif mode == 'eval_train':
            eval_dataset = cifar_dataset(dataset=self.dataset, noise_type=self.noise_type, cifarn_mode=self.cifarn_mode,
                                         r=self.r,
                                         root_dir=self.root_dir, transform=self.transform_test, mode='normal',
                                         noise_file=self.noise_file)
            eval_loader = DataLoader(
                dataset=eval_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)
            return eval_loader
