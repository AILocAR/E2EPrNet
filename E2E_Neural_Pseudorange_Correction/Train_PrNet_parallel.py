import torch
import theseus as th
import time
import statistics
import matplotlib.pyplot as plt
from torch import nn
from d2l import torch as d2l
from DataPreprocessing_PrNet_parallel_urban import data_preprocessing
from Differentiable_Localization_Layer import differentiable_localization_Theseus_layer
torch.set_default_tensor_type(torch.DoubleTensor)
# torch.autograd.set_detect_anomaly(True)


def train_gnss_net(net, data_iter, lr, num_epochs, device):
    """Train PrNet."""
    # Delegate computation to CPU or GPU
    net.to(device)

    # Count learnable parameters
    # for param in net.parameters():
    #     print(type(param), param.size())
    # print(sum([p.numel() for p in net.parameters()]))
    # assert 1==0

    # Determine the optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    
    # Define loss function
    loss = nn.MSELoss()
    # loss = nn.HuberLoss()

    # Set the neural network to training mode
    net.train()

    # Set figure to plot training loss
    animator = d2l.Animator(xlabel='epoch', ylabel='loss')

    # time_step = 0 

    # Evaluate training time
    time_sum = []

    # Training epoch by epoch
    for epoch in range(num_epochs):
        for batch in data_iter:           
            optimizer.zero_grad()
                        
            # time_step = time_step + 1

            # Read a batch of training data and delegate the data to our device
            # Shape of enc_x: (batch_size, PRN_size, input_size)
            enc_x, _ = [x.to(device) for x in batch]
            
            # Data Preprocessing
            # Shape of post_enc_x:       ('batch_size', 'PRN_size', 'input_feature_size')
            # Shape of valid_prn_index:  ('batch_size', 'PRN_size') 
            # Positions with true values mean the correseponding satellites are visiable
            post_enc_x, valid_prn_index = data_preprocessing(enc_x, device)

            start_time = time.time()
            # Pass input data through the neural network
            # Shape of dec_y_scaled:    ('batch_size', 'PRN_size', 1)
            # Shape of dec_y:           ('batch_size', 'PRN_size', 1)
            # Shape of total_prm_error: ('batch_size', 'PRN_size', 1)
            total_prm_error = net(post_enc_x)

            # **********************Inner Loop Differentiable localization layer**********************
            # The position of satellites
            sv_xyz = enc_x[:, :, 2:5]
            
            # The corrected pseudoranges = pseudorange + satellite clock bias - atmosphere delays
            prm_c = enc_x[:, :, 9:10] + enc_x[:, :, 5:6] - enc_x[:, :, 7:8]
            
            # Pseudorange weights
            # Shape of prUncertainty: ('batch_size', 'PRN_size')
            prUncertainty = enc_x[:, :, 17]
            prUncertainty[valid_prn_index] = 1/prUncertainty[valid_prn_index]
            # Shape of Wpr: ('batch_size', 'PRN_size', 'PRN_size')
            Wpr = torch.diag_embed(prUncertainty)

            # The WLS based location estimation
            # Shape of wls_xyz: ('batch_size', 'PRN_size', 3)
            # Shape of wls_t: ('batch_size', 'PRN_size', 1)
            wls_xyz = enc_x[:, :, 10:13]
            wls_t = enc_x[:, :, 13:14]
            
            # Theseus layer
            # Shape of hat_xyzt: batch size * 4
            hat_xyzt = differentiable_localization_Theseus_layer(total_prm_error, valid_prn_index, 
                                                                 sv_xyz, prm_c, Wpr,
                                                                 wls_xyz, wls_t,
                                                                 device)
                                    
            #  **********************Outer Loop Loss Computation**********************
            # Target values of user locations
            # Use ground truth from SPAN system
            # Shape of gt_xyz0: batch size * prn_size * 3
            gt_xyz0 = enc_x[:, :, 14:17]
            # Shape of gt_xyz: batch size * 3
            gt_xyz = torch.sum(gt_xyz0, dim=1)/torch.count_nonzero(gt_xyz0, dim=1)

            # Target values of user clock bias
            # Use WLS based estimation
            # Shape of gt_t0: batch size * prn_size * 1
            gt_t0 = enc_x[:, :, 13:14]

            # # Use RTS based estimation
            # # Shape of gt_t0: batch size * prn_size * 1
            # gt_t0 = enc_x[:, :, 65:66]

            # Shape of gt_t: batch size * 1
            gt_t = torch.sum(gt_t0, dim=1)/torch.count_nonzero(gt_t0, dim=1)

            # ## ************************************************************
            # # Weighted loss using state estimation uncertainty
            # # Shape of uncertaintyXyzt0: batch size * prn_size * 4
            # # Variance of x, y, z, dtu estimations
            # uncertaintyXyzt0 = enc_x[:, :, 69:73]

            # # Shape of uncertaintyXyzt: batch size * 4
            # uncertaintyXyzt = torch.sum(uncertaintyXyzt0, dim=1)/torch.count_nonzero(uncertaintyXyzt0, dim=1)

            # W_xyzt = 1/torch.pow(uncertaintyXyzt, 0.5)

            # weighted_est_xyzt = hat_xyzt * W_xyzt

            # weighted_gt_xyzt = torch.concat([gt_xyz, gt_t], dim=-1) * W_xyzt

            # J = loss(weighted_est_xyzt, weighted_gt_xyzt)
            # ## ************************************************************
            
            # J = loss(hat_xyzt, torch.concat([gt_xyz, gt_t], dim=-1))

            # No receiver clock offset
            J = loss(hat_xyzt[:, 0:3], gt_xyz)

            # Backward Gradient Descent
            J.sum().backward()

            # Gradient Clipping
            # if isinstance(net, nn.Module):
            #     params = [p for p in net.parameters() if p.requires_grad]
            # else:
            #     params = net.params
                        
            # # norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))            
            # for param in params:
            #     norm = param.grad ** 2
            #     if norm > 1:
            #         param.grad[:] *= 1 / norm

            nn.utils.clip_grad_norm_(net.parameters(), 100, norm_type=2)
            # d2l.grad_clipping(net, 1)
            optimizer.step()
            # animator.add(time_step, [J.cpu().detach().numpy()])
            
            # Count training time
            end_time = time.time()
            elapsed_time = end_time - start_time
            time_sum.append(elapsed_time)

        # Plot the training loss per epoch
        if (epoch + 1) % 1 == 0:
            animator.add(epoch + 1, [J.cpu().detach().numpy()])
            

        # if (epoch + 1) % 1 == 0:
        #     print('Epoch', epoch+1, 'is done. ', 'Loss is',J.cpu().detach().numpy())
        # if (epoch + 1) % 100 == 0:           
        #     filename = 'PrNet_2023V_' + str(epoch+1+1900) + '.tar'       
        #     torch.save({
        #         'model_state_dict': net.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         }, filename)
    elapsed_time_per_batch = sum(time_sum)/len(time_sum)
    print('Training time per batch: ', elapsed_time_per_batch)
    return optimizer