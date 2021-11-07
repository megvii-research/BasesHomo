import numpy as np
import torch
import torch.nn as nn
from model import net

def triplet_loss(a, p, n, margin=1.0, exp=1, reduce=False, size_average=False):

    triplet_loss = nn.TripletMarginLoss(margin=margin, p=exp, reduce=reduce, size_average=size_average)
    return triplet_loss(a, p, n)

def photo_loss_function(diff, q, averge=True):
    diff = (torch.abs(diff) + 0.01).pow(q)
    if averge:
        loss_mean = diff.mean()
    else:
        loss_mean = diff.sum()
    return loss_mean

def geometricDistance(correspondence, flow):
    flow = flow.permute(1, 2, 0).cpu().detach().numpy()

    p1 = correspondence[0] # 0
    p2 = correspondence[1] # 1

    if isinstance(correspondence[1][0], float):
        result = p2 - (p1 - flow[int(p1[1]), int(p1[0])])
        error = np.linalg.norm(result)
    else:
        result = [p2 - (p1 - flow[p1[1], p1[0]]), p1 - (p2 - flow[p2[1], p2[0]])]
        error = min(np.linalg.norm(result[0]), np.linalg.norm(result[1]))

    return error

def compute_losses(output, train_batch, params):
    losses = {}

    # compute losses
    if params.loss_type == "basic":
        imgs_patch = train_batch['imgs_gray_patch']
        start = train_batch['start']

        H_flow_f, H_flow_b = output['H_flow']
        fea1_full, fea2_full = output["fea_full"]
        fea1_patch, fea2_patch = output["fea_patch"]
        img1_warp, img2_warp = output["img_warp"]
        fea1_patch_warp, fea2_patch_warp = output["fea_patch_warp"]

        batch_size, _, h_patch, w_patch = imgs_patch.size()

        fea2_warp = net.get_warp_flow(fea2_full, H_flow_f, start=start)
        fea1_warp = net.get_warp_flow(fea1_full, H_flow_b, start=start)

        im_diff_fw = imgs_patch[:, :1, ...] - img2_warp
        im_diff_bw = imgs_patch[:, 1:, ...] - img1_warp

        fea_diff_fw = fea1_warp - fea1_patch_warp
        fea_diff_bw = fea2_warp - fea2_patch_warp

        # loss
        losses["photo_loss_l1"] = photo_loss_function(diff=im_diff_fw, q=1, averge=True) +  photo_loss_function(diff=im_diff_bw, q=1, averge=True)

        losses["fea_loss_l1"] = photo_loss_function(diff=fea_diff_fw, q=1, averge=True) +  photo_loss_function(diff=fea_diff_bw, q=1, averge=True)

        losses["triplet_loss"] = triplet_loss(fea1_patch, fea2_warp, fea2_patch).mean() +  triplet_loss(fea2_patch, fea1_warp, fea1_patch).mean()

        # loss toal: backward needed
        losses["total"] = losses["triplet_loss"]  + params.weight_fil * losses["fea_loss_l1"]

    else:
        raise NotImplementedError

    return losses

def compute_eval_results(data_batch, output_batch, manager):

    imgs_full = data_batch["imgs_ori"]
    points = data_batch["points"]
    batch_size, _, grid_h, grid_w = imgs_full.shape

    H_flow_f, H_flow_b = output_batch['H_flow']

    H_flow_f = net.upsample2d_flow_as(H_flow_f, imgs_full, mode="bilinear", if_rate=True)  # scale
    H_flow_b = net.upsample2d_flow_as(H_flow_b, imgs_full, mode="bilinear", if_rate=True)
    img1_full_warp = net.get_warp_flow(imgs_full[:, :3, ...], H_flow_b, start=0)
    img2_full_warp = net.get_warp_flow(imgs_full[:, 3:, ... ], H_flow_f, start=0)

    errs = []
    errs_p = []

    for i in range(len(points)):  # len(points)
        point = eval(points[i])
        err = 0
        tmp = []
        for j in range(6):  # len(point['matche_pts'])
            points_value = point['matche_pts'][j]
            err_p = geometricDistance(points_value, H_flow_f[i])
            err += err_p
            tmp.append(err_p)

        errs.append(err / (j + 1))
        errs_p.append(tmp)
    # ==================================================================== return ======================================================================

    eval_results = {}
    eval_results["img1_full_warp"] = img1_full_warp
    eval_results["errs"] = errs
    eval_results["errs_p"] = errs_p

    return eval_results
