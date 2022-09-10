import os
import time
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms, utils
from torch.autograd import Variable

from batch_transformers import BatchToTensor, BatchRGBToYCbCr, YCbCrToRGB
from dataset import MEFdataset
from model import DeepFuse
from mefssim import MEF_MSSSIM

EPS = 1e-8

class Trainer():
    def __init__(self, config):
        torch.manual_seed(config.seed)

        # self.transform = transforms.Compose([
        #     BatchToTensor()
        # ])
        self.transform = transforms.Compose([
            BatchToTensor(),
            BatchRGBToYCbCr()
        ])

        self.train_batch_size = 32
        self.test_batch_size = 1

        # training set configuration
        self.train_data = MEFdataset(data_path=config.trainset,
                                        transform=self.transform,
                                        phase='train')

        self.train_loader = DataLoader(self.train_data,
                                       batch_size=self.train_batch_size,
                                       shuffle=False,
                                       pin_memory=True,
                                       num_workers=1)

        # testing set configuration
        self.test_data = MEFdataset(data_path=config.testset,
                                        transform=self.transform,
                                        phase='test')

        self.test_loader = DataLoader(self.test_data,
                                      batch_size=self.test_batch_size,
                                      shuffle=False,
                                      pin_memory=True,
                                      num_workers=1)

        # initialize the model
        self.model = DeepFuse()
        self.model_name = type(self.model).__name__
        # print(self.model)

        # loss function
        self.loss_fn = MEF_MSSSIM(is_lum=True)
        self.initial_lr = config.lr
        if self.initial_lr is None:
            lr = 0.0005
        else:
            lr = self.initial_lr
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)


        if torch.cuda.is_available() and config.use_cuda:
            self.model.cuda()
            self.loss_fn = self.loss_fn.cuda()

        # some states
        self.start_epoch = 0
        self.start_step = 0
        self.train_loss = []
        self.test_results = []
        self.best_result = 0
        self.ckpt_path = config.ckpt_path
        self.use_cuda = config.use_cuda
        self.max_epochs = config.max_epochs
        # self.finetune_epochs = config.finetune_epochs
        # self.finetuneset = config.finetuneset
        self.epochs_per_eval = config.epochs_per_eval
        self.epochs_per_save = config.epochs_per_save
        self.fused_img_path = config.fused_img_path
        self.model_path = config.model_path

        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                             last_epoch=self.start_epoch-1,
                                             step_size=config.decay_interval,
                                             gamma=config.decay_ratio)

    def fit(self):
        for epoch in range(self.start_epoch, self.max_epochs):
            # print(epoch)
            # print(self.max_epochs - self.finetune_epochs - 1)
            # if epoch > self.max_epochs - self.finetune_epochs - 1:
            # _ = self._train_single_epoch(epoch)
            num_steps_per_epoch = len(self.train_loader)
            local_counter = epoch * num_steps_per_epoch + 1
            start_time = time.time()
            beta = 0.9
            running_loss = 0 if epoch == 0 else self.train_loss[-1]
            loss_corrected = 0.0
            running_duration = 0.0

            # start training
            for step, sample_batched in enumerate(self.train_loader, 0):
                # TODO: remove this after debugging
                im_over, im_under, im = sample_batched['hr_over'], sample_batched['hr_under'], sample_batched['hr']

                if self.train_batch_size > 1:
                    Y_over = im_over[:, 0, :, :].unsqueeze(1)
                    Y_under = im_under[:, 0, :, :].unsqueeze(1)
                    Y = im[:, 0, :, :].unsqueeze(1)

                else:
                    im_over = torch.squeeze(im_over, dim=0)
                    im_under = torch.squeeze(im_under, dim=0)
                    im = torch.squeeze(im, dim=0)

                    Y_over = im_over[0, :, :].unsqueeze(0)
                    Y_under = im_under[0, :, :].unsqueeze(0)
                    Y = im[0, :, :].unsqueeze(0)

                    Y_over = Y_over.unsqueeze(0)
                    Y_under = Y_under.unsqueeze(0)
                
                Y_over = Variable(Y_over)
                Y_under = Variable(Y_under)
                Y = Variable(Y)

                if self.use_cuda:
                    Y_over = Y_over.cuda()
                    Y_under = Y_under.cuda()
                    Y = Y.cuda()

                self.optimizer.zero_grad()


                self.model.setInput(Y_under, Y_over)
                Yf = self.model.forward()

                if self.train_batch_size > 1:
                    batch_loss = 0
                    for i in range(Yf.shape[0]):
                        Yf_i = Yf[i,:,:,:].unsqueeze(0)
                        Y_under_i = Y_under[i,:,:,:].unsqueeze(0)
                        Y_over_i = Y_over[i,:,:,:].unsqueeze(0)

                        batch_loss -= self.loss_fn(Yf_i, torch.cat((Y_under_i, Y_over_i), dim=0))

                    self.loss = batch_loss/Yf.shape[0]

                else:
                    self.loss = -self.loss_fn(Yf, torch.cat((Y_under, Y_over), dim=0))

                self.loss.backward()
                self.optimizer.step()

                q = -self.loss.data.item()

                # statistics
                running_loss = beta * running_loss + (1 - beta) * q
                loss_corrected = running_loss / (1 - beta ** local_counter)

                current_time = time.time()
                duration = current_time - start_time
                running_duration = beta * running_duration + (1 - beta) * duration
                duration_corrected = running_duration / (1 - beta ** local_counter)
                examples_per_sec = self.train_batch_size / duration_corrected
                format_str = ('(E:%d, S:%d) [MEF-SSIM = %.4f] (%.1f samples/sec; %.3f '
                            'sec/batch)')
                print(format_str % (epoch, step, loss_corrected,
                                    examples_per_sec, duration_corrected))

                local_counter += 1
                self.start_step = 0
                start_time = time.time()

            self.train_loss.append(loss_corrected)
            self.scheduler.step()

            if (epoch+1) % self.epochs_per_eval == 0:
                # evaluate after every other epoch
                test_results = self.eval(epoch)
                self.test_results.append(test_results)

                state = {
                    'model': self.model.state_dict(),
                    'loss': self.train_loss
                }

                if test_results > self.best_result:
                    torch.save(state, os.path.join(self.model_path, 'best_ep.pth'))
                    self.best_result = test_results

                fig_val = plt.figure()
                plt.plot(self.test_results)
                plt.savefig('val_SSIM_curve.png')
                plt.close()

                out_str = 'Epoch {} Testing: Average MEF-SSIM: {:.4f}'.format(epoch, test_results)
                print(out_str)
            
            if (epoch+1) % self.epochs_per_save == 0:
                model_name = '{}-{:0>5d}.pt'.format(self.model_name, epoch)
                model_name = os.path.join(self.ckpt_path, model_name)
                self._save_checkpoint({
                    'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'train_loss': self.train_loss,
                    'test_results': self.test_results,
                }, model_name)

            matplotlib.use('Agg')
            fig_train = plt.figure()
            plt.plot(self.train_loss)
            plt.savefig('train_loss_curve.png')
            plt.close()

    def _train_single_epoch(self, epoch):
        # initialize logging system
        num_steps_per_epoch = len(self.train_loader)
        local_counter = epoch * num_steps_per_epoch + 1
        start_time = time.time()
        beta = 0.9
        running_loss = 0 if epoch == 0 else self.train_loss[-1]
        loss_corrected = 0.0
        running_duration = 0.0

        # start training
        # print('Adam learning rate: {:f}'.format(self.optimizer.param_groups[0]['lr']))
        for step, sample_batched in enumerate(self.train_loader, 0):
            # TODO: remove this after debugging
            im_over, im_under, im = sample_batched['hr_over'], sample_batched['hr_under'], sample_batched['hr']

            # print(im_over.shape)

            if self.train_batch_size > 1:
                Y_over = im_over[:, 0, :, :].unsqueeze(1)
                Y_under = im_under[:, 0, :, :].unsqueeze(1)
                Y = im[:, 0, :, :].unsqueeze(1)

            else:
                im_over = torch.squeeze(im_over, dim=0)
                im_under = torch.squeeze(im_under, dim=0)
                im = torch.squeeze(im, dim=0)

                # print(im_over.permute(1, 2, 0).shape)
                # plt.imshow(im_over.permute(1, 2, 0))
                # plt.show()
                # time.sleep(5)

                Y_over = im_over[0, :, :].unsqueeze(0)
                Y_under = im_under[0, :, :].unsqueeze(0)
                Y = im[0, :, :].unsqueeze(0)

                Y_over = Y_over.unsqueeze(0)
                Y_under = Y_under.unsqueeze(0)
            
            Y_over = Variable(Y_over)
            Y_under = Variable(Y_under)
            Y = Variable(Y)

            if self.use_cuda:
                Y_over = Y_over.cuda()
                Y_under = Y_under.cuda()
                Y = Y.cuda()

            self.optimizer.zero_grad()

            # print(Y_over.shape)
            # print(Y_under.shape)

            self.model.setInput(Y_under, Y_over)
            Yf = self.model.forward()

            # print(Yf.shape)

            if self.train_batch_size > 1:
                batch_loss = 0
                for i in range(Yf.shape[0]):
                    Yf_i = Yf[i,:,:,:].unsqueeze(0)
                    Y_under_i = Y_under[i,:,:,:].unsqueeze(0)
                    Y_over_i = Y_over[i,:,:,:].unsqueeze(0)

                    batch_loss -= self.loss_fn(Yf_i, torch.cat((Y_under_i, Y_over_i), dim=0))

                self.loss = batch_loss/Yf.shape[0]

            else:
                self.loss = -self.loss_fn(Yf, torch.cat((Y_under, Y_over), dim=0))

            self.loss.backward()
            self.optimizer.step()

            q = -self.loss.data.item()

            # statistics
            running_loss = beta * running_loss + (1 - beta) * q
            loss_corrected = running_loss / (1 - beta ** local_counter)

            current_time = time.time()
            duration = current_time - start_time
            running_duration = beta * running_duration + (1 - beta) * duration
            duration_corrected = running_duration / (1 - beta ** local_counter)
            examples_per_sec = self.train_batch_size / duration_corrected
            format_str = ('(E:%d, S:%d) [MEF-SSIM = %.4f] (%.1f samples/sec; %.3f '
                          'sec/batch)')
            print(format_str % (epoch, step, loss_corrected,
                                examples_per_sec, duration_corrected))

            local_counter += 1
            self.start_step = 0
            start_time = time.time()

        self.train_loss.append(loss_corrected)
        self.scheduler.step()

        if (epoch+1) % self.epochs_per_eval == 0:
            # evaluate after every other epoch
            test_results = self.eval(epoch)
            self.test_results.append(test_results)
            out_str = 'Epoch {} Testing: Average MEF-SSIM: {:.4f}'.format(epoch, test_results)
            print(out_str)

        if (epoch+1) % self.epochs_per_save == 0:
            model_name = '{}-{:0>5d}.pt'.format(self.model_name, epoch)
            model_name = os.path.join(self.ckpt_path, model_name)
            self._save_checkpoint({
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'train_loss': self.train_loss,
                'test_results': self.test_results,
            }, model_name)

        return self.loss.data.item()

    def eval(self, epoch):
        scores = []
        for step, sample_batched in enumerate(self.test_loader, 0):
            # TODO: remove this after debugging
            im_over, im_under = sample_batched['hr_over'], sample_batched['hr_under']

            im_over = torch.squeeze(im_over, dim=0)
            im_under = torch.squeeze(im_under, dim=0)

            if 'hr' in sample_batched:
                im = sample_batched['hr']
                im = torch.squeeze(im, dim=0)

            im_exp = torch.stack((im_under, im_over), dim=0)

            Ys = im_exp[:, 0, :, :].unsqueeze(1)
            Cbs = im_exp[:, 1, :, :].unsqueeze(1)
            Crs = im_exp[:, 2, :, :].unsqueeze(1)

            Wb = (torch.abs(Cbs - 0.5) + EPS) / torch.sum(torch.abs(Cbs - 0.5) + EPS, dim=0)
            Wr = (torch.abs(Crs - 0.5) + EPS) / torch.sum(torch.abs(Crs - 0.5) + EPS, dim=0)

            Cb_f = torch.sum(Wb * Cbs, dim=0, keepdim=True).clamp(0, 1)
            Cr_f = torch.sum(Wr * Crs, dim=0, keepdim=True).clamp(0, 1)

            # Y_over = im_over[0, :, :].unsqueeze(0)
            # Cb_over = im_over[1, :, :].unsqueeze(0)
            # Cr_over = im_over[2, :, :].unsqueeze(0)

            # Y_under = im_over[0, :, :].unsqueeze(0)
            # Cb_under = im_over[1, :, :].unsqueeze(0)
            # Cr_under = im_over[2, :, :].unsqueeze(0)
            
            Y_under = im_under[0, :, :].unsqueeze(0)
            Y_over = im_over[0, :, :].unsqueeze(0)
            # Y = im[0, :, :].unsqueeze(0)

            Y_over = Variable(Y_over)
            Y_under = Variable(Y_under)
            # Y = Variable(Y)

            if self.use_cuda:
                Y_over = Y_over.cuda()
                Y_under = Y_under.cuda()
                # Y = Y.cuda()
                Ys = Ys.cuda()

            Y_over = Y_over.unsqueeze(0)
            Y_under = Y_under.unsqueeze(0)

            with torch.no_grad():
                self.model.setInput(Y_under, Y_over)
                Yf = self.model.forward()

                q = self.loss_fn(Yf, Ys).cpu()
            
            scores.append(q.data.numpy())

            Yf_RGB = YCbCrToRGB()(torch.cat((Yf.cpu(), Cb_f, Cr_f), dim=1))
            self._save_image(Yf_RGB, self.fused_img_path, str(epoch) + '_' + str(step))

        avg_quality = sum(scores) / len(scores)
        return avg_quality

    # save checkpoint
    @staticmethod
    def _save_checkpoint(state, filename='checkpoint.pth.tar'):
        # if os.path.exists(filename):
        #     shutil.rmtree(filename)
        torch.save(state, filename)

    def _save_image(self, image, path, name):
        b = image.size()[0]
        for i in range(b):
            t = image.data[i]
            t[t > 1] = 1
            t[t < 0] = 0
            utils.save_image(t, "%s/%s_%d.png" % (path, name, i))