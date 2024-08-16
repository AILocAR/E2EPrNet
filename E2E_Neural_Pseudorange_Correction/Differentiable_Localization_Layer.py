import torch
import theseus as th
from d2l import torch as d2l
torch.set_default_tensor_type(torch.DoubleTensor)
# torch.autograd.set_detect_anomaly(True)
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = d2l.try_gpu()

# Cost function for pseudorange localization
def prm_error_fn(optim_vars, aux_vars):
    xyz, t = optim_vars
    total_prm_error, sv_xyz, prm_c, Wpr= aux_vars 
    
    # # Shape of valid_prn_index: batch size * PRN_size *1
    # valid_prn_index = (sv_xyz[:, :, 1] != 0).unsqueeze(dim = -1)   
    # # Shape of xyz.tensor: batch size * 3
    # # Shape of sv_xyz.tensor: batch size * PRN_size * 3    
    # # Construct a tensor with shape (batch size * PRN size * 3) with location and time info for visible satellites
    # th_xyz = torch.zeros(sv_xyz.tensor.size(), device=d2l.try_gpu())
    # th_xyz[valid_prn_index.repeat(1, 1, 3)] = xyz.tensor.unsqueeze(dim=1).repeat(1,sv_xyz.tensor.size(dim=1), 1)[valid_prn_index.repeat(1, 1, 3)]
    # th_t = torch.zeros(sv_xyz.tensor.size(dim=0), sv_xyz.tensor.size(dim=1), 1, device=d2l.try_gpu())
    # th_t[valid_prn_index] = t.tensor.unsqueeze(dim=1).repeat(1,sv_xyz.tensor.size(dim=1), 1)[valid_prn_index]
    # # Shape of estimation: batch size * PRN_size * 1
    # estimation = torch.linalg.vector_norm(th_xyz - sv_xyz.tensor, dim = -1, keepdim = True)+th_t
    # # Shape of error: batch size * PRN_size * 1
    # error = estimation - (prm_c.tensor+total_prm_error.tensor)

    # Shape of xyz.tensor: batch size * 3
    # Shape of t.tensor: batch size * 1
    # Shape of estimation: batch size * PRN_size * 1
    estimation = torch.linalg.vector_norm(xyz.tensor.unsqueeze(dim=1) - sv_xyz.tensor, dim = -1, keepdim = True)+t.tensor.unsqueeze(dim=1)
    error = estimation - (prm_c.tensor+total_prm_error.tensor)
   
    # Shape of Wpr: batch size * PRN_size * PRN_size
    weighted_error = torch.bmm(Wpr.tensor, error)
    # Shape of weighted_error: batch size * PRN_size
    return weighted_error.squeeze(dim=-1)


# ------------------------Differentiable localization layer powered by Theseus------------------------
# Inputs: 
# #1 data_total_prm_error: neural pseudorange corrections with shape ('batch_size', 'PRN_size', 1)
# #2 valid_prn_index: visible satellite index with shape ('batch_size', 'PRN_size')
# the positions with the value of True are visible
# #3 data_sv_xyz: satellite locations with shape ('batch_size', 'PRN_size', 3)
# #4 data_prm_c: corrected pseudorange measurements with shape ('batch_size', 'PRN_size', 1)
# #5 data_Wpr: pseudorange weights with shape ('batch_size', 'PRN_size', 'PRN_size')
# #6 wls_xyz, wls_t: initial values with shape ('batch_size', 'PRN_size', 3) and ('batch_size', 'PRN_size', 1)
# Outputs:
# hat_user: User state estimations including x, y,z, delta t_u with shape  ('batch_size', 4)
# ----------------------------------------------------------------------------------------------------
def differentiable_localization_Theseus_layer(data_total_prm_error, data_valid_prn_index, 
                                              data_sv_xyz, data_prm_c, data_Wpr,
                                              wls_xyz, wls_t, device):
    # Step 1: Construct optimization and auxiliary Theseus variables.
    # Construct variables of the function: these the optimization variables of the cost functions.
    # Unknown states include location x,y,z and receiver clock bias delta t
    # xyz is initialized as a zero tensor with shape batch size * 3 
    # t is initialized as a zero tensor with shape batch size * 1
    xyz = th.Vector(tensor = torch.zeros(data_valid_prn_index.size(dim=0), 3).to(device=device), name="xyz")  
    t = th.Vector(tensor = torch.zeros(data_valid_prn_index.size(dim=0), 1).to(device=device), name="t")

    # Construct auxiliary variables.
    total_prm_error = th.Variable(data_total_prm_error, name="total_prm_error")
    sv_xyz = th.Variable(data_sv_xyz, name="sv_xyz")
    prm_c = th.Variable(data_prm_c, name="prm_c")
    valid_prn_index = th.Variable(data_valid_prn_index, name="valid_prn_index")
   
    # Step 2: Construct cost weights
    Wpr = th.Variable(data_Wpr, name="Wpr")

    # Step 3: Declare optimization and aunxiliary variables
    optim_vars = xyz, t
    aux_vars = total_prm_error, sv_xyz, prm_c, Wpr

    # Step 4: Construct cost functions
    cost_function = th.AutoDiffCostFunction(optim_vars, prm_error_fn, 
                                            data_valid_prn_index.size(dim=1), 
                                            aux_vars=aux_vars, name="pseudorange_cost_fn")
  
    # Step 5: Create the objective, optimizer and Theseus layer    
    objective = th.Objective()
    objective.add(cost_function)
    optimizer = th.GaussNewton(objective, 
                               th.CholeskyDenseSolver,
                               max_iterations=50, 
                               step_size=0.5,)

#     optimizer = th.GaussNewton(objective, 
#                                th.CholmodSparseSolver,
#                                max_iterations=50, 
#                                step_size=0.5,)
    # optimizer = th.GaussNewton(objective, 
    #                            th.LUCudaSparseSolver,
    #                            max_iterations=50, 
    #                            step_size=0.5,)
    # optimizer = th.GaussNewton(objective, 
    #                             max_iterations=50, 
    #                             step_size=0.5,)
    theseus_optim = th.TheseusLayer(optimizer)
    theseus_optim.to(device=device)
    

    # Step 6: Run Theseus layer
    # Initialize the input variables of Theseus layer
    # Shape of xyz0: batch size * 3
    xyz0 = torch.sum(wls_xyz, dim=1)/torch.count_nonzero(wls_xyz, dim=1)
    # Shape of t0: batch size * 1
    t0 = torch.sum(wls_t, dim=1)/torch.count_nonzero(wls_t, dim=1)
    theseus_inputs = {}
    theseus_inputs.update({
            "total_prm_error": data_total_prm_error,
            "sv_xyz": data_sv_xyz,
            "prm_c": data_prm_c,
            "Wpr": data_Wpr,
            "xyz": xyz0,
            "t": t0
            })
    
    optimizer_kwargs = {}
    optimizer_kwargs.update({
            "backward_mode": "UNROLL"
    })
#     optimizer_kwargs.update({
#             "backward_mode": "DLM",
#             "dlm_epsilon": 5e-6
#     })
    # optimizer_kwargs.update({
    #         "backward_mode": "TRUNCATED",
    #         "backward_num_iterations": 5
    # })

    updated_inputs, _ = theseus_optim.forward(theseus_inputs, optimizer_kwargs)
    return torch.concat([updated_inputs["xyz"], updated_inputs["t"]], dim=-1)
    


    
    

