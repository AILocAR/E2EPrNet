import torch
import numpy as np
import time
import coordinates as coord
from torch import nn
from DataPreprocessing_PrNet_parallel import data_preprocessing
from Differentiable_Localization_Layer import differentiable_localization_Theseus_layer
from d2l import torch as d2l
from tqdm import tqdm

torch.set_default_tensor_type(torch.DoubleTensor)
# torch.autograd.set_detect_anomaly(True)


def evaluate_gnss_net(net, data_iter, batch_size, device):
    """Inference using PrFormer"""

    # Delegate computation to CPU or GPU
    net.to(device)

    # Set the neural network to training mode
    net.eval()

    # Define loss function
    loss = nn.MSELoss()

    # Set figure to plot inference loss
    animator = d2l.Animator(xlabel='time step', ylabel='loss')

    time_step = 0

    time_sum = []

    # Initialize the output list
    output_seq = []
    gt_output_seq = []
    prm_output_seq = []

    # batch size is 1
    for batch in data_iter:

        time_step = time_step+1

        # Read a batch of training data and delegate the data to our device
        # Shape of enc_x: (batch_size, PRN_size, input_size)
        enc_x, _ = [z.to(device) for z in batch]

        # Data Preprocessing
        # Shape of post_enc_x:       ('batch_size', 'PRN_size', 'input_feature_size')
        # Shape of valid_prn_index:  ('batch_size', 'PRN_size')
        post_enc_x, valid_prn_index = data_preprocessing(enc_x, device)

        start_time = time.time()

        # Pass input data through the neural network
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

        # Inference time
        end_time = time.time()
        elapsed_time = end_time - start_time
        time_sum.append(elapsed_time)

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
        # Shape of gt_t: batch size * 1
        gt_t = torch.sum(gt_t0, dim=1)/torch.count_nonzero(gt_t0, dim=1)
        
        J = loss(hat_xyzt[:, 0:3], gt_xyz)

        Jt = loss(hat_xyzt[:, 3:4], gt_t)

        # Form the output of location and time estimations
        for i in range(batch_size):
            ## Form location and time estimations
            # Shape of enc_x_per_batch: ('PRN_size')
            enc_x_per_batch = enc_x[i, :, 0] 

            # Shape of prm_per_batch: ('PRN_size', input_size)
            prm_per_batch = enc_x[i]

            # Shape of epoch_index: (1)
            epoch_index = torch.sum(enc_x_per_batch)/torch.count_nonzero(enc_x_per_batch)

            # Convert ECEF coordinates to geodetic coordinates (lon, lat, h)
            # Shape of lla: 1*3
            lla = coord.ecef2geodetic(hat_xyzt[i:i+1, 0:3].cpu().detach().numpy())
            
            # Shape of delta_t_u: 1*1
            delta_t_u = hat_xyzt[i:i+1, 3:4].cpu().detach().numpy()
            
            # Shape of index_prn_xyzt_per_batch: 1*5
            index_xyzt_per_batch = np.concatenate((epoch_index.reshape(1, 1).cpu().detach().numpy(), lla, delta_t_u), axis=1)

            # # Shape of index_prn_xyzt_per_batch: (1, 3)
            # index_prn_xyzt_per_batch = torch.cat([epoch_index.reshape(1, 1), hat_xyzt[i:i+1, :]], dim=1)
            output_seq.append(index_xyzt_per_batch)

            ## Form location and time ground truth
            # Shape of gt_lla_per_batch: 1*3 (lon, lat, h)
            gt_lla = coord.ecef2geodetic(gt_xyz[i:i+1, :].cpu().detach().numpy())

            # Shape of gt_delta_t_u: 1*1
            gt_delta_t_u = gt_t[i:i+1, :].cpu().detach().numpy()

            # Shape of index_prn_xyzt_per_batch: 1*5
            index_gt_xyzt_per_batch = np.concatenate((epoch_index.reshape(1, 1).cpu().detach().numpy(), gt_lla, gt_delta_t_u), axis=1)
            gt_output_seq.append(index_gt_xyzt_per_batch)

            ## Form pseudorange corrections and labels
            # Shape of index_prn_prmbias_per_batch: (`Valid_PRN_size`, 5): epoch, prn, neural pseudorange correction, noisy pseudorange errors, smoothed pseudorange errors
            index_prn_prmbias_per_batch = torch.cat([prm_per_batch[valid_prn_index[i], 0:2], -total_prm_error[i,valid_prn_index[i],:], prm_per_batch[valid_prn_index[i], 31:32], prm_per_batch[valid_prn_index[i], 34:35]], dim=1)
            prm_output_seq.append(index_prn_prmbias_per_batch)

        animator.add(time_step, [J.cpu().detach().numpy()])

    elapsed_time_per_sample = sum(time_sum)/len(time_sum)
    print('Inference time per sample: ', elapsed_time_per_sample)
    # Shape of 'return': (`All_epochs`, 4)
    return  np.concatenate(output_seq, axis=0), np.concatenate(gt_output_seq, axis=0), torch.cat(prm_output_seq, dim=0)


