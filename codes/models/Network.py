import torch
import torch.nn as nn
import utils.utils as utils 
import os
import numpy as np
import logging
import random
from models.CIN import CIN
from models.modules.loss import ReconstructionImgLoss, ReconstructionMsgLoss, EncodedLoss
import torch.optim.lr_scheduler as lr
import utils.checkpoint as check
import kornia



# ---------------------------------------------------------------------------------
class Network(nn.Module):
    def __init__(self, opt, device, path_in):
        super(Network, self).__init__()
        # 
        self.opt = opt
        self.device = device
        self.device_ids = opt['train']['device_ids']
        # init model
        self.cinNet = CIN(opt, self.device)
        # parallel
        self.cinNet = nn.DataParallel(self.cinNet, device_ids=self.device_ids) 
        # init checkpoint
        self.Checkpoint = check.Checkpoint(path_in['path_checkpoint'], opt)
        # loading resume state if exists
        if opt['train']['resume']['Empty'] != True:
            self.resume()
        # loss
        self.RecImgLoss  = ReconstructionImgLoss(opt)
        self.RecMsgLoss  = ReconstructionMsgLoss(opt)
        self.encodedLoss = EncodedLoss(opt)
        # start_lr
        self.Lr_current = opt['lr']['start_lr']
        # optimizer
        if opt['lr']['optimizer'] == 'Adam':
            self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.cinNet.parameters()), lr=self.Lr_current)
        elif opt['lr']['optimizer'] == 'SGD':
            self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.cinNet.parameters()), lr=self.Lr_current, momentum=0.9)
        # lr scheduler
        self.lr_milestones = opt['lr']['milestones']
        self.lr_gamma = opt['lr']['gamma']
        self.scheduler = lr.MultiStepLR(self.optimizer, milestones=self.lr_milestones, gamma=self.lr_gamma)
        # step config
        self.current_step = 0
        # loss weight
        self.lw = utils.loss_lamd(self.current_step, opt)    
        # path
        self.img_w_folder_tra =  path_in['img_w_folder_tra']
        self.loss_w_folder = path_in['loss_w_folder']
        self.log_folder = path_in['log_folder']
        self.img_w_folder_val = path_in['img_w_folder_val']
        self.img_w_folder_test = path_in['img_w_folder_test']
        self.time_now_NewExperiment = path_in['time_now_NewExperiment']


    def train(self, train_data, current_epoch):
        logging.info('--------------------------------------------------------\n')
        logging.info('##### train #####\n')
        self.current_epoch = current_epoch
        #
        with torch.enable_grad():
            #
            loss_per_epoch_sum = 0
            loss_per_epoch_msg = 0
            loss_per_epoch_enc = 0
            train_step = 0
            for _, image in enumerate(train_data):  
                #
                self.cinNet.train()
                self.optimizer.zero_grad()
                #
                image = image.to(self.device)  
                # msg
                if self.opt['datasets']['msg']['mod_a']:     # msg in [0, 1]
                    message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], self.opt['network']['message_length']))).to(self.device)
                elif self.opt['datasets']['msg']['mod_b']:   # msg in [-1, 1]
                    message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], self.opt['network']['message_length']))).to(self.device)
                    message = message * 2 - 1
                else:
                    print('input message error, exit!')
                    exit()

                # fix noise-layer each batch
                if self.opt['noise']['option'] == 'Combined':
                    NoiseName = self.opt['noise']['Combined']['names']
                    noise_choice = random.choice(NoiseName)   
                else:
                    noise_choice = self.opt['noise']['option']
                # cinNet
                watermarking_img, noised_img, img_fake, msg_fake_1, msg_fake_2, _ = \
                    self.cinNet(image, message, noise_choice, True)
                # loss
                loss_RecImg  =  self.RecImgLoss(image, img_fake)
                loss_RecMsg  =  utils.func_loss_RecMsg(self.RecMsgLoss, message, msg_fake_1, msg_fake_2)
                loss_encoded =  self.encodedLoss(image, watermarking_img)   

                # total loss
                train_loss = utils.func_loss(self.lw, loss_RecImg, loss_encoded, loss_RecMsg)
                train_loss.backward()
                self.optimizer.step()

                # log print
                if self.current_step % self.opt['train']['logs_per_step'] == 0:
                    psnr_wm2co,psnr_no2co,psnr_rec2co,psnr_wm2no,ssim_wm2co,ssim_rec2co,acc1,BitWise_AvgErr1,\
                        acc2,BitWise_AvgErr2 = self.psnr_ssim_acc(image, watermarking_img,noised_img,img_fake,msg_fake_1,msg_fake_2,message)
                    #
                    utils.log_info(self.lw, self.current_epoch, self.opt['train']['epoch'], self.current_step, self.Lr_current, psnr_wm2co, psnr_no2co, psnr_rec2co, psnr_wm2no, \
                            ssim_wm2co, ssim_rec2co, BitWise_AvgErr1, BitWise_AvgErr2, train_loss, loss_RecImg, loss_RecMsg, loss_encoded, noise_choice)
                
                # save images
                if  self.current_step % self.opt["train"]['saveTrainImgs_per_step'] == 0:
                    utils.mkdir(self.img_w_folder_tra)
                    utils.save_images(image, watermarking_img, noised_img, img_fake, self.current_epoch, self.current_step, self.img_w_folder_tra, self.time_now_NewExperiment, self.opt, resize_to=None)

                # break code if 'nan'
                if torch.isnan(train_loss):
                    logging.info("Invalid loss <nan>, break code !")
                    exit()

                # step update
                self.current_step += 1
                train_step += 1
                loss_per_epoch_sum = loss_per_epoch_sum + (loss_RecMsg.item() + loss_encoded.item())
                loss_per_epoch_msg = loss_per_epoch_msg + loss_RecMsg.item()
                loss_per_epoch_enc = loss_per_epoch_enc + loss_encoded.item()
                
                # loss weight update
                self.lw = utils.loss_lamd(self.current_step, self.opt)
                
                # lr update
                self.scheduler.step()
                self.Lr_current = self.scheduler.get_last_lr()[0]

            # Checkpoint
            if self.current_epoch % self.opt['train']['checkpoint_per_epoch'] == 0:
                logging.info('Checkpoint: Saving cinNets and training states.')
                self.Checkpoint.save(self.cinNet, self.current_step, self.current_epoch, 'cinNet')

            # write losses
            utils.mkdir(self.loss_w_folder)
            utils.write_losses(os.path.join(self.loss_w_folder, 'train-{}.txt'.format(self.time_now_NewExperiment)), self.current_epoch, loss_per_epoch_sum/train_step, loss_per_epoch_msg/train_step, loss_per_epoch_enc/train_step)
        

    def validation(self, val_data, current_epoch):
        #
        logging.info('--------------------------------------------------------\n')
        logging.info('##### validation #####\n')
        self.current_epoch = current_epoch
        #
        val_step = 0
        psnr_wm2no_mean = 0
        psnr_wm2co_mean = 0
        psnr_rec2co_mean = 0
        psnr_no2co_mean = 0
        BitWise_AvgErr1_mean = 0
        BitWise_AvgErr2_mean = 0
        ssim_wm2co_mean = 0
        ssim_rec2co_mean = 0
        #
        loss_per_val_sum = 0
        loss_per_val_msg = 0
        loss_per_val_enc = 0
        #
        with torch.no_grad():
            # 
            for _, image in enumerate(val_data):  # torch.utils.data.Dataset
                #
                self.cinNet.eval()
                image = image.to(self.device)
                # msg
                if self.opt['datasets']['msg']['mod_a']:     # msg in [0, 1]
                    message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], self.opt['network']['message_length']))).to(self.device)
                elif self.opt['datasets']['msg']['mod_b']:   # msg in [-1, 1]
                    message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], self.opt['network']['message_length']))).to(self.device)
                    message = message * 2 - 1
                else:
                    print('input message error, exit!')
                    exit()                                

                # fix noise-layer each batch
                if self.opt['noise']['option'] == 'Combined':
                    NoiseName = self.opt['noise']['Combined']['names']
                    noise_choice = random.choice(NoiseName)
                else:
                    noise_choice = self.opt['noise']['option']
                # cinNet
                watermarking_img, noised_img, img_fake, msg_fake_1, msg_fake_2, _ = \
                    self.cinNet(image, message, noise_choice, True)
                # loss
                loss_RecImg  =  self.RecImgLoss(image, img_fake)
                loss_RecMsg =  utils.func_loss_RecMsg(self.RecMsgLoss, message, msg_fake_1, msg_fake_2)
                loss_encoded =  self.encodedLoss(image, watermarking_img)   
                val_loss = loss_RecMsg + loss_encoded

                # log print
                if val_step % self.opt['train']['val']['logs_per_step'] == 0:
                    psnr_wm2co,psnr_no2co,psnr_rec2co,psnr_wm2no,ssim_wm2co,ssim_rec2co,acc1,BitWise_AvgErr1,\
                        acc2,BitWise_AvgErr2 = self.psnr_ssim_acc(image, watermarking_img,noised_img,img_fake,msg_fake_1,msg_fake_2,message)
                    #
                    utils.log_info(self.lw, self.current_epoch, 0, self.current_step, self.Lr_current, psnr_wm2co, psnr_no2co, psnr_rec2co, psnr_wm2no, \
                            ssim_wm2co, ssim_rec2co, BitWise_AvgErr1, BitWise_AvgErr2, val_loss, loss_RecImg, loss_RecMsg, loss_encoded, noise_choice)
                
                # save images
                if  val_step == self.opt["train"]['saveValImgs_in_step']:
                    utils.mkdir(self.img_w_folder_val)
                    utils.save_images(image, watermarking_img, noised_img, img_fake, self.current_epoch, self.current_step, self.img_w_folder_val, self.time_now_NewExperiment, self.opt, resize_to=None)
                
                # loss write
                val_step += 1
                loss_per_val_sum = loss_per_val_sum + (loss_RecMsg.item() + loss_encoded.item())
                loss_per_val_msg = loss_per_val_msg + loss_RecMsg.item()
                loss_per_val_enc = loss_per_val_enc + loss_encoded.item()

                # mean
                psnr_wm2no_mean = psnr_wm2no_mean + psnr_wm2no.item()
                psnr_wm2co_mean = psnr_wm2co_mean + psnr_wm2co.item()
                psnr_rec2co_mean = psnr_rec2co_mean + psnr_rec2co.item()
                psnr_no2co_mean = psnr_no2co_mean + psnr_no2co.item()
                #
                ssim_wm2co_mean = ssim_wm2co_mean + ssim_wm2co.item()
                ssim_rec2co_mean =  ssim_rec2co_mean + ssim_rec2co.item()

                #
                BitWise_AvgErr1_mean    = utils.func_mean_filter_None(BitWise_AvgErr1_mean, BitWise_AvgErr1, 'add')
                BitWise_AvgErr2_mean    = utils.func_mean_filter_None(BitWise_AvgErr2_mean, BitWise_AvgErr2, 'add')

            # logging mean psnr
            print_BitWise_AvgErr1_mean    = utils.func_mean_filter_None(BitWise_AvgErr1_mean, val_step, 'div')
            print_BitWise_AvgErr2_mean    = utils.func_mean_filter_None(BitWise_AvgErr2_mean, val_step, 'div')

            # logging mean 
            logging.info('\npsnr_no2co_mean = {}\n\
                            psnr_rec2co_mean = {}\n\
                            psnr_wm2no_mean = {}\n\
                            psnr_wm2co_mean = {}\n \
                            #------------------------\n\
                            BitWise_AvgErr_1 = {}\n\
                            BitWise_AvgErr_2 = {}\n\
                            ------------------------\n\
                            ssim_wm2co_mean = {}\n\
                            ssim_rec2co_mean = {}'\
                    .format(psnr_no2co_mean/val_step, \
                            psnr_rec2co_mean/val_step, \
                            psnr_wm2no_mean/val_step, \
                            psnr_wm2co_mean/val_step, \
                            #------------------------
                            print_BitWise_AvgErr1_mean, \
                            print_BitWise_AvgErr2_mean, \
                            #------------------------
                            ssim_wm2co_mean/val_step, \
                            ssim_rec2co_mean/val_step))
                    
            #
            utils.mkdir(self.loss_w_folder)
            utils.write_losses(os.path.join(self.loss_w_folder, 'val-{}.txt'.format(self.time_now_NewExperiment)), self.current_epoch, loss_per_val_sum/val_step, loss_per_val_msg/val_step, loss_per_val_enc/val_step)
        #
        logging.info('--------------------------------------------------------\n')


    # def test(self, test_data, current_epoch):  
    #     #
    #     logging.info('--------------------------------------------------------\n')
    #     logging.info('##### test only #####\n')
    #     self.current_epoch = current_epoch


    #     log_file = os.path.join(self.log_folder, f'test_results_epoch_{current_epoch}.txt')
    
    #     # Write header to log file
    #         with open(log_file, 'w') as f:
    #         f.write(f"Test Results - Epoch {current_epoch}\n")
    #         f.write("="*80 + "\n")
    #         f.write("Image_ID, PSNR_WM2CO, PSNR_NO2CO, PSNR_REC2CO, PSNR_WM2NO, SSIM_WM2CO, SSIM_REC2CO, BER, ACC, BitWise_AvgErr1, BitWise_AvgErr2, Noise_Type\n")
    #         f.write("-"*80 + "\n")

    #     #
    #     with torch.no_grad():
    #         #
    #         self.cinNet.eval()
    #         test_step = 0 
    #         total_steps = len(test_data)
    #         #
    #         psnr_wm2no_mean = 0  
    #         psnr_wm2co_mean = 0
    #         psnr_rec2co_mean = 0
    #         psnr_no2co_mean = 0
    #         BER_mean = 0
    #         ssim_wm2co_mean = 0
    #         ssim_rec2co_mean = 0

    #         total_images = 0

    #         # 
    #         for _, image in enumerate(test_data):  
    #             #
    #             if (test_step*self.opt['train']['batch_size']) >= self.opt['datasets']['test']['num']:
    #                 break
    #             #
    #             image = image.to(self.device)
    #             batch_size = image.shape[0]

    #             # msg
    #             if self.opt['datasets']['msg']['mod_a']:     # msg in [0, 1]
    #                 message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], self.opt['network']['message_length']))).to(self.device)
    #             elif self.opt['datasets']['msg']['mod_b']:   # msg in [-1, 1]
    #                 message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], self.opt['network']['message_length']))).to(self.device)
    #                 message = message * 2 - 1
    #             else:
    #                 print('input message error, exit!')
    #                 exit()

    #             # fix noise-layer each batch
    #             if self.opt['noise']['option'] == 'Combined':
    #                 NoiseName = self.opt['noise']['Combined']['names']
    #                 noise_choice = random.choice(NoiseName)   
    #             else:
    #                 noise_choice = self.opt['noise']['option']
    #             # cinNet
    #             watermarking_img, noised_img, img_fake, msg_fake_1, msg_fake_2, msg_nsm = \
    #                 self.cinNet(image, message, noise_choice, False)
    #             # psnr ssim acc
    #             psnr_wm2co,psnr_no2co,psnr_rec2co,psnr_wm2no,ssim_wm2co,ssim_rec2co,acc1,BitWise_AvgErr1,\
    #                 acc2,BitWise_AvgErr2 = self.psnr_ssim_acc(image, watermarking_img,noised_img,img_fake,msg_fake_1,msg_fake_2,message)

    #             # mean
    #             psnr_wm2no_mean = psnr_wm2no_mean + psnr_wm2no
    #             psnr_wm2co_mean = psnr_wm2co_mean + psnr_wm2co
    #             psnr_rec2co_mean = psnr_rec2co_mean + psnr_rec2co
    #             psnr_no2co_mean = psnr_no2co_mean + psnr_no2co
    #             #
    #             ssim_wm2co_mean = ssim_wm2co.item() + ssim_wm2co_mean
    #             ssim_rec2co_mean = ssim_rec2co.item() + ssim_rec2co_mean
    #             #
    #             _, ber = utils.bitWise_accurary(msg_nsm, message, self.opt)
    #             BER_mean = BER_mean + ber

    #             # loggging
    #             if  test_step % self.opt["train"]['logTest_per_step'] == 0:
    #                 utils.log_info_test(test_step, total_steps, self.Lr_current, psnr_wm2co, psnr_no2co, psnr_rec2co, psnr_wm2no,\
    #                     ssim_wm2co, ssim_rec2co, BitWise_AvgErr1, BitWise_AvgErr2, noise_choice)
                
    #             # save images
    #             if  test_step % self.opt["train"]['saveTestImgs_per_step'] == 0:
    #                 utils.mkdir(self.img_w_folder_test)
    #                 utils.save_images(image, watermarking_img, noised_img, img_fake, self.current_epoch, test_step, self.img_w_folder_test, self.time_now_NewExperiment, self.opt, resize_to=None)
    #             #
    #             test_step += 1

    #         # logging mean 
    #         logging.info('\n\
    #                         psnr_no2co_mean = {}\n\
    #                         psnr_rec2co_mean = {}\n\
    #                         psnr_wm2no_mean = {}\n\
    #                         psnr_wm2co_mean = {}\n \
    #                         ------------------------\n\
    #                         BER_ave = {}\n\
    #                         ACC_ave = {}\n\
    #                         ------------------------\n\
    #                         ssim_wm2co_mean = {}\n\
    #                         ssim_rec2co_mean = {}'\
    #                 .format(psnr_no2co_mean.item()/test_step, \
    #                         psnr_rec2co_mean.item()/test_step, \
    #                         psnr_wm2no_mean.item()/test_step, \
    #                         psnr_wm2co_mean.item()/test_step, \
    #                         #------------------------
    #                         BER_mean/test_step, \
    #                         (1-BER_mean/test_step)*100, \
    #                         #------------------------
    #                         ssim_wm2co_mean/test_step, \
    #                         ssim_rec2co_mean/test_step))
    #     # test 1 epoch
    #     logging.info('--------------------------------------------------------\n')
    #     logging.info('Test end !')

    def test(self, test_data, current_epoch):  
    #
        logging.info('--------------------------------------------------------\n')
        logging.info('##### test only #####\n')
        self.current_epoch = current_epoch
        
        # Create log file for individual image results
        log_file = os.path.join(self.log_folder, f'test_results_epoch_{current_epoch}.txt')
        
        # Write header to log file
        with open(log_file, 'w') as f:
            f.write(f"Test Results - Epoch {current_epoch}\n")
            f.write("="*80 + "\n")
            f.write("Image_ID, PSNR_WM2CO, PSNR_NO2CO, PSNR_REC2CO, PSNR_WM2NO, SSIM_WM2CO, SSIM_REC2CO, BER, ACC, BitWise_AvgErr1, BitWise_AvgErr2, Noise_Type\n")
            f.write("-"*80 + "\n")
        
        #
        with torch.no_grad():
            #
            self.cinNet.eval()
            test_step = 0 
            total_steps = len(test_data)
            #
            psnr_wm2no_mean = 0  
            psnr_wm2co_mean = 0
            psnr_rec2co_mean = 0
            psnr_no2co_mean = 0
            BER_mean = 0
            ssim_wm2co_mean = 0
            ssim_rec2co_mean = 0
            
            total_images = 0
            
            # 
            for _, image in enumerate(test_data):  
                #
                if (test_step*self.opt['train']['batch_size']) >= self.opt['datasets']['test']['num']:
                    break
                #
                image = image.to(self.device)
                batch_size = image.shape[0]
                
                # msg
                if self.opt['datasets']['msg']['mod_a']:     # msg in [0, 1]
                    message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], self.opt['network']['message_length']))).to(self.device)
                elif self.opt['datasets']['msg']['mod_b']:   # msg in [-1, 1]
                    message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], self.opt['network']['message_length']))).to(self.device)
                    message = message * 2 - 1
                else:
                    print('input message error, exit!')
                    exit()

                # fix noise-layer each batch
                if self.opt['noise']['option'] == 'Combined':
                    NoiseName = self.opt['noise']['Combined']['names']
                    noise_choice = random.choice(NoiseName)   
                else:
                    noise_choice = self.opt['noise']['option']
                # cinNet
                watermarking_img, noised_img, img_fake, msg_fake_1, msg_fake_2, msg_nsm = \
                    self.cinNet(image, message, noise_choice, False)
                
                # Process each image in the batch individually
                for i in range(batch_size):
                    image_idx = total_images + i
                    
                    # Extract individual images and messages
                    single_image = image[i:i+1]
                    single_watermarking_img = watermarking_img[i:i+1]
                    single_noised_img = noised_img[i:i+1]
                    single_img_fake = img_fake[i:i+1]
                    single_msg_fake_1 = msg_fake_1[i:i+1] if msg_fake_1 is not None else None
                    single_msg_fake_2 = msg_fake_2[i:i+1] if msg_fake_2 is not None else None
                    single_msg_nsm = msg_nsm[i:i+1] if msg_nsm is not None else None
                    single_message = message[i:i+1] 
                    
                    # Calculate metrics for this single image
                    psnr_wm2co, psnr_no2co, psnr_rec2co, psnr_wm2no, ssim_wm2co, ssim_rec2co, acc1, BitWise_AvgErr1, acc2, BitWise_AvgErr2 = \
                        self.psnr_ssim_acc(single_image, single_watermarking_img, single_noised_img, single_img_fake, single_msg_fake_1, single_msg_fake_2, single_message)
                    
                    # Calculate BER for this single image
                    _, ber = utils.bitWise_accurary(single_msg_nsm, single_message, self.opt)
                    acc = 1 - ber
                    
                    # Log individual image results
                    log_entry = f"{image_idx:04d}, {psnr_wm2co.item():.6f}, {psnr_no2co.item():.6f}, {psnr_rec2co.item():.6f}, {psnr_wm2no.item():.6f}, {ssim_wm2co.item():.6f}, {ssim_rec2co.item():.6f}, {ber:.6f}, {acc:.6f}"
                    
                    if BitWise_AvgErr1 is not None:
                        log_entry += f", {BitWise_AvgErr1:.6f}"
                    else:
                        log_entry += ", N/A"
                    
                    if BitWise_AvgErr2 is not None:
                        log_entry += f", {BitWise_AvgErr2:.6f}"
                    else:
                        log_entry += ", N/A"
                    
                    log_entry += f", {noise_choice}\n"
                    
                    # Write to log file
                    with open(log_file, 'a') as f:
                        f.write(log_entry)
                    
                    # Print to console for immediate feedback (each image)
                    print(f"Image {image_idx:04d}: PSNR_WM2CO={psnr_wm2co.item():.6f}, PSNR_NO2CO={psnr_no2co.item():.6f}, PSNR_REC2CO={psnr_rec2co.item():.6f}, PSNR_WM2NO={psnr_wm2no.item():.6f}, SSIM_WM2CO={ssim_wm2co.item():.6f}, SSIM_REC2CO={ssim_rec2co.item():.6f}, BER={ber:.6f}, ACC={acc:.6f}, Noise={noise_choice}")
                    
                    # Accumulate metrics for each individual image
                    psnr_wm2no_mean += psnr_wm2no.item()
                    psnr_wm2co_mean += psnr_wm2co.item()
                    psnr_rec2co_mean += psnr_rec2co.item()
                    psnr_no2co_mean += psnr_no2co.item()
                    ssim_wm2co_mean += ssim_wm2co.item()
                    ssim_rec2co_mean += ssim_rec2co.item()
                    BER_mean += ber
                    
                    # Save individual image if needed
                    if image_idx % self.opt["train"]['saveTestImgs_per_step'] == 0:
                        utils.mkdir(self.img_w_folder_test)
                        utils.save_images(single_image, single_watermarking_img, single_noised_img, single_img_fake, 
                                        self.current_epoch, image_idx, self.img_w_folder_test, self.time_now_NewExperiment, self.opt, resize_to=None)
                    
                    # Update total images count
                    total_images += 1
                    
                    # Print running averages every 10 images
                    if total_images % 10 == 0:
                        print(f"--- Running Average (after {total_images} images) ---")
                        print(f"Avg PSNR_WM2CO: {psnr_wm2co_mean/total_images:.6f}")
                        print(f"Avg PSNR_NO2CO: {psnr_no2co_mean/total_images:.6f}")
                        print(f"Avg PSNR_REC2CO: {psnr_rec2co_mean/total_images:.6f}")
                        print(f"Avg PSNR_WM2NO: {psnr_wm2no_mean/total_images:.6f}")
                        print(f"Avg BER: {BER_mean/total_images:.6f}")
                        print(f"Avg ACC: {(1-BER_mean/total_images)*100:.2f}%")
                        print(f"Avg SSIM_WM2CO: {ssim_wm2co_mean/total_images:.6f}")
                        print(f"Avg SSIM_REC2CO: {ssim_rec2co_mean/total_images:.6f}")
                        print("-" * 50)
                
                # Log batch progress (for compatibility with existing logging)
                if test_step % self.opt["train"]['logTest_per_step'] == 0:
                    # Use batch-level metrics for progress logging
                    batch_psnr_wm2co, batch_psnr_no2co, batch_psnr_rec2co, batch_psnr_wm2no, batch_ssim_wm2co, batch_ssim_rec2co, batch_acc1, batch_BitWise_AvgErr1, batch_acc2, batch_BitWise_AvgErr2 = \
                        self.psnr_ssim_acc(image, watermarking_img, noised_img, img_fake, msg_fake_1, msg_fake_2, message)
                    
                    utils.log_info_test(test_step, total_steps, self.Lr_current, batch_psnr_wm2co, batch_psnr_no2co, batch_psnr_rec2co, batch_psnr_wm2no,
                        batch_ssim_wm2co, batch_ssim_rec2co, batch_BitWise_AvgErr1, batch_BitWise_AvgErr2, noise_choice)
                
                test_step += 1

            # Write average results to log file
            with open(log_file, 'a') as f:
                f.write("\n" + "="*80 + "\n")
                f.write("FINAL AVERAGE RESULTS:\n")
                f.write("-"*80 + "\n")
                f.write(f"Total Images Processed: {total_images}\n")
                f.write(f"Average PSNR_NO2CO: {psnr_no2co_mean/total_images:.6f}\n")
                f.write(f"Average PSNR_REC2CO: {psnr_rec2co_mean/total_images:.6f}\n")
                f.write(f"Average PSNR_WM2NO: {psnr_wm2no_mean/total_images:.6f}\n")
                f.write(f"Average PSNR_WM2CO: {psnr_wm2co_mean/total_images:.6f}\n")
                f.write(f"Average BER: {BER_mean/total_images:.6f}\n")
                f.write(f"Average ACC: {(1-BER_mean/total_images)*100:.6f}%\n")
                f.write(f"Average SSIM_WM2CO: {ssim_wm2co_mean/total_images:.6f}\n")
                f.write(f"Average SSIM_REC2CO: {ssim_rec2co_mean/total_images:.6f}\n")
                f.write("="*80 + "\n")

            # Print final averages to console
            print("\n" + "="*80)
            print("FINAL AVERAGE RESULTS:")
            print("-"*80)
            print(f"Total Images Processed: {total_images}")
            print(f"Average PSNR_NO2CO: {psnr_no2co_mean/total_images:.6f}")
            print(f"Average PSNR_REC2CO: {psnr_rec2co_mean/total_images:.6f}")
            print(f"Average PSNR_WM2NO: {psnr_wm2no_mean/total_images:.6f}")
            print(f"Average PSNR_WM2CO: {psnr_wm2co_mean/total_images:.6f}")
            print(f"Average BER: {BER_mean/total_images:.6f}")
            print(f"Average ACC: {(1-BER_mean/total_images)*100:.2f}%")
            print(f"Average SSIM_WM2CO: {ssim_wm2co_mean/total_images:.6f}")
            print(f"Average SSIM_REC2CO: {ssim_rec2co_mean/total_images:.6f}")
            print("="*80)

            # logging mean (console output - for compatibility)
            logging.info('\n\
                            psnr_no2co_mean = {}\n\
                            psnr_rec2co_mean = {}\n\
                            psnr_wm2no_mean = {}\n\
                            psnr_wm2co_mean = {}\n \
                            ------------------------\n\
                            BER_ave = {}\n\
                            ACC_ave = {}\n\
                            ------------------------\n\
                            ssim_wm2co_mean = {}\n\
                            ssim_rec2co_mean = {}'\
                    .format(psnr_no2co_mean/total_images, \
                            psnr_rec2co_mean/total_images, \
                            psnr_wm2no_mean/total_images, \
                            psnr_wm2co_mean/total_images, \
                            #------------------------
                            BER_mean/total_images, \
                            (1-BER_mean/total_images)*100, \
                            #------------------------
                            ssim_wm2co_mean/total_images, \
                            ssim_rec2co_mean/total_images))
        
        # test 1 epoch
        logging.info('--------------------------------------------------------\n')
        logging.info(f'Test end! Individual results saved to: {log_file}')
        
        return {
            'psnr_wm2co': psnr_wm2co_mean/total_images,
            'psnr_no2co': psnr_no2co_mean/total_images,
            'psnr_rec2co': psnr_rec2co_mean/total_images,
            'psnr_wm2no': psnr_wm2no_mean/total_images,
            'ber': BER_mean/total_images,
            'acc': (1-BER_mean/total_images)*100,
            'ssim_wm2co': ssim_wm2co_mean/total_images,
            'ssim_rec2co': ssim_rec2co_mean/total_images,
            'total_images': total_images
        }

    def psnr_ssim_acc(self, image, watermarking_img, noised_img, img_fake, msg_fake_1, msg_fake_2, message):
        # psnr
        psnr_wm2co = kornia.metrics.psnr(
            ((image + 1) / 2).clamp(0, 1),
            ((watermarking_img.detach() + 1) / 2).clamp(0, 1),
            1,
        )
        psnr_no2co = kornia.metrics.psnr(
            ((image + 1) / 2).clamp(0, 1),
            ((noised_img.detach() + 1) / 2).clamp(0, 1),
            1,
        )
        psnr_rec2co = kornia.metrics.psnr(
            ((image + 1) / 2).clamp(0, 1),
            ((img_fake.detach() + 1) / 2).clamp(0, 1),
            1,
        )
        psnr_wm2no = kornia.metrics.psnr(
            ((image + 1) / 2).clamp(0, 1),
            ((noised_img.detach() + 1) / 2).clamp(0, 1),
            1,
        )
        # ssim
        ssim_wm2co = kornia.metrics.ssim(
            ((image + 1) / 2).clamp(0, 1),
            ((watermarking_img.detach() + 1) / 2).clamp(0, 1),
            window_size=11,
        ).mean()
        ssim_rec2co = kornia.metrics.ssim(
            ((image + 1) / 2).clamp(0, 1),
            ((img_fake.detach() + 1) / 2).clamp(0, 1),
            window_size=11,
        ).mean()
        # acc
        acc1, BitWise_AvgErr1 = utils.bitWise_accurary(msg_fake_1, message, self.opt)
        acc2, BitWise_AvgErr2 = utils.bitWise_accurary(msg_fake_2, message, self.opt)
        return psnr_wm2co, psnr_no2co, psnr_rec2co, psnr_wm2no, ssim_wm2co, ssim_rec2co, acc1,BitWise_AvgErr1, acc2, BitWise_AvgErr2
        

    def resume(self):
        # resuming: all param loaded into default GPU
        device_id = torch.cuda.current_device()
        if self.opt['train']['resume']['one_pth']:
             # resume CIN.IM and CIN.NIAM, and NSM
            resume_state = torch.load(self.opt['path']['resume_state_1pth'], map_location=lambda storage, loc: storage.cuda(device_id))
            self.cinNet = self.Checkpoint.resume_training(self.cinNet, 'cinNet', resume_state)  
        else:
            # resume CIN.IM and CIN.NIAM
            resume_state = torch.load(self.opt['path']['resume_state_cinNet'], map_location=lambda storage, loc: storage.cuda(device_id))
            self.cinNet = self.Checkpoint.resume_training(self.cinNet, 'model_state_dict', resume_state)  
            # resume CIN.NSM
            resume_state_nsmNet = torch.load(self.opt['path']['resume_state_nsmNet'], map_location=lambda storage, loc: storage.cuda(device_id))
            self.cinNet.module.nsm_model = self.Checkpoint.resume_training(self.cinNet.module.nsm_model, 'nsmNet', resume_state_nsmNet)  
