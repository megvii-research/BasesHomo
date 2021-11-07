from __future__ import absolute_import, division, print_function
import numpy as np
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import warnings
from torch.nn.modules.utils import _pair, _quadruple

warnings.filterwarnings("ignore")
 
class Net(nn.Module):

    def __init__(self, params):

        super(Net, self).__init__()
        self.inplanes = 64
        self.layers = [3, 4, 6, 3]
        self.basis_vector_num = 16
        self.crop_size = params.crop_size
        self.basis = gen_basis(self.crop_size[0], self.crop_size[1]).unsqueeze(0).reshape(1, 8, -1) # 8,2,h,w --> 1, 8, 2*h*w

        self.share_feature = ShareFeature(1)
        self.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.LeakyReLU(inplace=True)
        self.block = BasicBlock
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(self.block, 64, self.layers[0])
        self.layer2 = self._make_layer(self.block, 128, self.layers[1], stride=2)
        self.layer3 = self._make_layer(self.block, 256, self.layers[2], stride=2)
        self.sp_layer3 = Subspace(256)
        self.layer4 = self._make_layer(self.block, 512, self.layers[3], stride=2)
        self.sp_layer4 = Subspace(512)

        self.subspace_block = SubspaceBlock(2, self.basis_vector_num)

        self.conv_last = nn.Conv2d(512, 8, kernel_size=1, stride=1, padding=0, groups=8, bias=False)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):

        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, input):
        # parse input
        x1_patch, x2_patch = input['imgs_gray_patch'][:, :1, ...], input['imgs_gray_patch'][:, 1:, ...]
        x1_full, x2_full = input["imgs_gray_full"][:, :1, ...],input["imgs_gray_full"][:, 1:, ...]
        start = input['start']

        batch_size, _, h_patch, w_patch = x1_patch.size()

        fea1_patch, fea2_patch = self.share_feature(x1_patch), self.share_feature(x2_patch)

        x = torch.cat([fea1_patch, fea2_patch], dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.sp_layer3(self.layer3(x))
        x = self.sp_layer4(self.layer4(x))
        x = self.conv_last(x) # bs,8,h,w
        weight_f = self.pool(x).squeeze(3) # bs,8,1,1
        H_flow_f = (self.basis * weight_f).sum(1).reshape(batch_size, 2, h_patch, w_patch)

        x = torch.cat([fea2_patch, fea1_patch], dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.sp_layer3(self.layer3(x))
        x = self.sp_layer4(self.layer4(x))
        x = self.conv_last(x)
        weight_b = self.pool(x).squeeze(3)  # bs,8,1,1
        H_flow_b = (self.basis * weight_b).sum(1).reshape(batch_size, 2, h_patch, w_patch)

        fea1_full, fea2_full = self.share_feature(x1_full), self.share_feature(x2_full)

        img1_warp = get_warp_flow(x1_full, H_flow_b, start=start)
        img2_warp = get_warp_flow(x2_full, H_flow_f, start=start)

        fea1_patch_warp, fea2_patch_warp = self.share_feature(img1_warp), self.share_feature(img2_warp)

        output = {}
        output['H_flow'] = [H_flow_f, H_flow_b]
        output['fea_full'] = [fea1_full, fea2_full]
        output["fea_patch"] = [fea1_patch, fea2_patch]
        output["fea_patch_warp"] = [fea1_patch_warp, fea2_patch_warp]
        output["img_warp"] = [img1_warp, img2_warp]
        output['basis_weight'] =[weight_f, weight_b]

        # import pdb
        # pdb.set_trace()
        
        return output

# ========================================================================================================================

class Subspace(nn.Module):

    def __init__(self, ch_in, k=16, use_SVD=True, use_PCA=False):

        super(Subspace, self).__init__()
        self.k = k
        self.Block = SubspaceBlock(ch_in, self.k)
        self.use_SVD = use_SVD
        self.use_PCA = use_PCA

    def forward(self, x):

        sub = self.Block(x)
        x = subspace_project(x, sub)

        return x

class SubspaceBlock(nn.Module):

    def __init__(self, inplanes, planes):

        super(SubspaceBlock, self).__init__()

        self.relu = nn.LeakyReLU(inplace=True)

        self.conv0 = conv(inplanes, planes, kernel_size=1, stride=1, dilation=1, isReLU=False)
        self.bn0 = nn.BatchNorm2d(planes)
        self.conv1 = conv(planes, planes, kernel_size=1, stride=1, dilation=1, isReLU=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv(planes, planes, kernel_size=1, stride=1, dilation=1, isReLU=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = self.relu(self.bn0(self.conv0(x)))

        out = self.conv1(residual)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual

        return out

class BasicBlock(nn.Module):

    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):

        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ShareFeature(nn.Module):

    def __init__(self, num_chs):

        super(ShareFeature, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(num_chs, 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),

            nn.Conv2d(4, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),

            nn.Conv2d(8, 1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.layers(x)
        return x

# ========================================================================================================================
# Some functions that are not used here, which are designed for homography computation, may be helpful in some other works.

def conv(in_planes, out_planes, kernel_size=3, stride=1, dilation=1, isReLU=True):

    if isReLU:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                      padding=((kernel_size - 1) * dilation) // 2, bias=True),
            nn.LeakyReLU(0.1, inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                      padding=((kernel_size - 1) * dilation) // 2, bias=True)
        )

def initialize_msra(modules):

    logging.info("Initializing MSRA")
    for layer in modules:
        if isinstance(layer, nn.Conv2d):
            nn.init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

        elif isinstance(layer, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

        elif isinstance(layer, nn.LeakyReLU):
            pass

        elif isinstance(layer, nn.Sequential):
            pass

def subspace_project(input, vectors):

    b_, c_, h_, w_ = input.shape
    basis_vector_num = vectors.shape[1]
    V_t = vectors.view(b_, basis_vector_num, h_ * w_)
    V_t = V_t / (1e-6 + V_t.abs().sum(dim=2, keepdim=True))
    V = V_t.permute(0, 2, 1)
    mat = torch.bmm(V_t, V)
    mat_inv = torch.inverse(mat)
    project_mat = torch.bmm(mat_inv, V_t)
    input_ = input.view(b_, c_, h_ * w_)
    project_feature = torch.bmm(project_mat, input_.permute(0, 2, 1))
    output = torch.bmm(V, project_feature).permute(0, 2, 1).view(b_, c_, h_, w_)

    return output

def gen_basis(h, w, qr=True, scale=True):

    N = 8
    d_range = 10
    a_range = 0.2
    t_range = 0.5
    p_range = 0.001
    grid = get_grid(1, h, w).permute(0, 2, 3, 1)
    flows = grid[:, :, :, :2] * 0

    for i in range(N):
        # initial H matrix
        dx, dy, ax, ay, px, py, tx, ty = 0, 0, 0, 0, 0, 0, 1, 1

        if i == 0:
            dx = d_range
        if i == 1:
            dy = d_range
        if i == 2:
            ax = a_range
        if i == 3:
            ay = a_range
        if i == 4:
            tx = t_range
        if i == 5:
            ty = t_range
        if i == 6:
            px = p_range
            fm = 1 # grid[:, :, :, 0] * px + 1
            grid_ = grid.clone().float()
            grid_[:, :, :, 0] = grid_[:, :, :, 0] ** 2 * px / fm
            grid_[:, :, :, 1] = grid[:, :, :, 0] * grid[:, :, :, 1] * px / fm
            flow = grid_[:, :, :, :2]

        elif i == 7:
            py = p_range
            fm = 1 # grid[:, :, :, 1] * py + 1
            grid_ = grid.clone().float()
            grid_[:, :, :, 1] = grid_[:, :, :, 1] ** 2  * py / fm
            grid_[:, :, :, 0] = grid[:, :, :, 0] * grid[:, :, :, 1] * py / fm
            flow = grid_[:, :, :, :2]
        else:
            H_mat = torch.tensor([[tx, ax, dx],
                                  [ay, ty, dy],
                                  [px, py, 1]]).float()
            H_mat = H_mat.cuda() if torch.cuda.is_available() else H_mat
            # warp grids
            H_mat = H_mat.unsqueeze(0).repeat(h * w, 1, 1).unsqueeze(0)
            grid_ = grid.reshape(-1, 3).unsqueeze(0).unsqueeze(3).float()  # shape: 3, h*w
            grid_warp = torch.matmul(H_mat, grid_)
            grid_warp = grid_warp.squeeze().reshape(h, w, 3).unsqueeze(0)

            flow = grid[:, :, :, :2] - grid_warp[:, :, :, :2] / grid_warp[:, :, :, 2:]
        flows = torch.cat((flows, flow), 0)

    flows = flows[1:, ...]
    if qr:
        flows_ = flows.reshape(8, -1).permute(1, 0)  # N, h, w, c --> N, h*w*c --> h*w*c, N
        flows_q, _ = torch.qr(flows_)
        flows_q = flows_q.permute(1, 0).reshape(8, h, w, 2)
        flows = flows_q

    if scale:
        max_value = flows.abs().reshape(8, -1).max(1)[0].reshape(8, 1, 1, 1)
        flows = flows / max_value

    return flows.permute(0,3,1,2) # 8,h,w,2-->8,2,h,w

def get_grid(batch_size, H, W, start=0):

    if torch.cuda.is_available():
        xx = torch.arange(0, W).cuda()
        yy = torch.arange(0, H).cuda()
    else:
        xx = torch.arange(0, W)
        yy = torch.arange(0, H)
    xx = xx.view(1, -1).repeat(H, 1)
    yy = yy.view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(batch_size, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(batch_size, 1, 1, 1)
    ones = torch.ones_like(xx).cuda() if torch.cuda.is_available() else torch.ones_like(xx)
    grid = torch.cat((xx, yy, ones), 1).float()

    grid[:, :2, :, :] = grid[:, :2, :, :] + start  # add the coordinate of left top
    return grid

def get_src_p(batch_size, patch_size_h, patch_size_w, divides, axis_t=False):

    small_gap_sz = [patch_size_h // divides, patch_size_w // divides]
    mesh_num = divides + 1
    if torch.cuda.is_available():
        xx = torch.arange(0, mesh_num).cuda()
        yy = torch.arange(0, mesh_num).cuda()
    else:
        xx = torch.arange(0, mesh_num)
        yy = torch.arange(0, mesh_num)

    xx = xx.view(1, -1).repeat(mesh_num, 1)
    yy = yy.view(-1, 1).repeat(1, mesh_num)
    xx = xx.view(1, 1, mesh_num, mesh_num) * small_gap_sz[1]
    yy = yy.view(1, 1, mesh_num, mesh_num) * small_gap_sz[0]
    xx[:, :, :, -1] = xx[:, :, :, -1] - 1
    yy[:, :, -1, :] = yy[:, :, -1, :] - 1
    if axis_t:
        ones = torch.ones_like(xx).cuda() if torch.cuda.is_available() else torch.ones_like(xx)
        src_p = torch.cat((xx, yy, ones), 1).repeat(batch_size, 1, 1, 1).float()
    else:
        src_p = torch.cat((xx, yy), 1).repeat(batch_size, 1, 1, 1).float()

    return src_p

def chunk_2D(img, h_num, w_num, h_dim=2, w_dim=3):

    bs, c, h, w = img.shape
    img = img.chunk(h_num, h_dim)
    img = torch.cat(img, dim=w_dim)
    img = img.chunk(h_num * w_num, w_dim)
    return torch.cat(img, dim=1).reshape(bs, c, h_num, w_num, h // h_num, w // w_num)

def get_point_pairs(src_p, divide):  # src_p: shape=(bs, 2, h, w)

    bs = src_p.shape[0]
    src_p = src_p.repeat_interleave(2, axis=2).repeat_interleave(2, axis=3)
    src_p = src_p[:, :, 1:-1, 1:-1]
    src_p = chunk_2D(src_p, divide, divide).reshape(bs, -1, 2, 2, 2)
    src_p = src_p.permute(0, 1, 3, 4, 2).reshape(bs, divide * divide, 4, 2)
    return src_p

def DLT_solve(src_p, off_set):

    bs, _, divide = src_p.shape[:3]
    divide = divide - 1

    src_ps = get_point_pairs(src_p, divide)
    off_sets = get_point_pairs(off_set, divide)

    bs, n, h, w = src_ps.shape
    N = bs * n

    src_ps = src_ps.reshape(N, h, w)
    off_sets = off_sets.reshape(N, h, w)

    dst_p = src_ps + off_sets

    ones = torch.ones(N, 4, 1).cuda() if torch.cuda.is_available() else torch.ones(N, 4, 1)
    xy1 = torch.cat((src_ps, ones), axis=2)
    zeros = torch.zeros_like(xy1).cuda() if torch.cuda.is_available() else torch.zeros_like(xy1)
    xyu, xyd = torch.cat((xy1, zeros), axis=2), torch.cat((zeros, xy1), axis=2)

    M1 = torch.cat((xyu, xyd), axis=2).reshape(N, -1, 6)
    M2 = torch.matmul(
        dst_p.reshape(-1, 2, 1),
        src_ps.reshape(-1, 1, 2),
    ).reshape(N, -1, 2)

    A = torch.cat((M1, -M2), axis=2)
    b = dst_p.reshape(N, -1, 1)

    Ainv = torch.inverse(A)

    h8 = torch.matmul(Ainv, b).reshape(N, 8)

    H = torch.cat((h8, ones[:, 0, :]), axis=1).reshape(N, 3, 3)
    H = H.reshape(bs, n, 3, 3)
    return H

def get_flow(H_mat_mul, patch_indices, patch_size_h, patch_size_w, divide, point_use=False):

    batch_size = H_mat_mul.shape[0]
    small_gap_sz = [patch_size_h // divide, patch_size_w // divide]

    small = 1e-7

    H_mat_pool = H_mat_mul.reshape(batch_size, divide, divide, 3, 3)  # .transpose(2,1)
    H_mat_pool = H_mat_pool.repeat_interleave(small_gap_sz[0], axis=1).repeat_interleave(small_gap_sz[1], axis=2)

    if point_use and H_mat_pool.shape[2] != patch_indices.shape[2]:
        H_mat_pool = H_mat_pool.permute(0, 3, 4, 1, 2)
        H_mat_pool = F.pad(H_mat_pool, pad=(0, 1, 0, 1, 0, 0), mode="replicate")
        H_mat_pool = H_mat_pool.permute(0, 3, 4, 1, 2)

    pred_I2_index_warp = patch_indices.permute(0, 2, 3, 1).unsqueeze(4)  # 把bs, 2, h, w 换为 bs, h, w, 2

    pred_I2_index_warp = torch.matmul(H_mat_pool, pred_I2_index_warp).squeeze(-1).permute(0, 3, 1, 2)
    T_t = pred_I2_index_warp[:, 2:3, ...]
    smallers = 1e-6 * (1.0 - torch.ge(torch.abs(T_t), small).float())
    T_t = T_t + smallers
    v1 = pred_I2_index_warp[:, 0:1, ...]
    v2 = pred_I2_index_warp[:, 1:2, ...]
    v1 = v1 / T_t
    v2 = v2 / T_t
    pred_I2_index_warp = torch.cat((v1, v2), 1)
    vgrid = patch_indices[:, :2, ...]

    flow = pred_I2_index_warp - vgrid
    return flow, vgrid

def transformer(I, vgrid, train=True):
    # I: Img, shape: batch_size, 1, full_h, full_w
    # vgrid: vgrid, target->source, shape: batch_size, 2, patch_h, patch_w
    # outsize: (patch_h, patch_w)

    def _interpolate(im, x, y, out_size):
        # x: x_grid_flat
        # y: y_grid_flat
        num_batch, num_channels, height, width = im.size()

        out_height, out_width = out_size[0], out_size[1]
        zero = 0
        max_y = height - 1
        max_x = width - 1

        # do sampling
        x0 = torch.floor(x).int()
        x1 = x0 + 1
        y0 = torch.floor(y).int()
        y1 = y0 + 1

        x0 = torch.clamp(x0, zero, max_x)  # same as np.clip
        x1 = torch.clamp(x1, zero, max_x)
        y0 = torch.clamp(y0, zero, max_y)
        y1 = torch.clamp(y1, zero, max_y)

        dim1 = width * height
        dim2 = width

        if torch.cuda.is_available():
            base = torch.arange(0, num_batch).int().cuda()
        else:
            base = torch.arange(0, num_batch).int()

        base = base * dim1
        base = base.repeat_interleave(out_height * out_width, axis=0)
        base_y0 = base + y0 * dim2
        base_y1 = base + y1 * dim2
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        # use indices to lookup pixels in the flat image and restore
        # channels dim
        im = im.permute(0, 2, 3, 1)
        im_flat = im.reshape([-1, num_channels]).float()

        idx_a = idx_a.unsqueeze(-1).long()
        idx_a = idx_a.expand(out_height * out_width * num_batch, num_channels)
        Ia = torch.gather(im_flat, 0, idx_a)

        idx_b = idx_b.unsqueeze(-1).long()
        idx_b = idx_b.expand(out_height * out_width * num_batch, num_channels)
        Ib = torch.gather(im_flat, 0, idx_b)

        idx_c = idx_c.unsqueeze(-1).long()
        idx_c = idx_c.expand(out_height * out_width * num_batch, num_channels)
        Ic = torch.gather(im_flat, 0, idx_c)

        idx_d = idx_d.unsqueeze(-1).long()
        idx_d = idx_d.expand(out_height * out_width * num_batch, num_channels)
        Id = torch.gather(im_flat, 0, idx_d)

        # and finally calculate interpolated values
        x0_f = x0.float()
        x1_f = x1.float()
        y0_f = y0.float()
        y1_f = y1.float()

        wa = torch.unsqueeze(((x1_f - x) * (y1_f - y)), 1)
        wb = torch.unsqueeze(((x1_f - x) * (y - y0_f)), 1)
        wc = torch.unsqueeze(((x - x0_f) * (y1_f - y)), 1)
        wd = torch.unsqueeze(((x - x0_f) * (y - y0_f)), 1)
        output = wa * Ia + wb * Ib + wc * Ic + wd * Id

        return output

    def _transform(I, vgrid):

        C_img = I.shape[1]
        B, C, H, W = vgrid.size()

        x_s_flat = vgrid[:, 0, ...].reshape([-1])
        y_s_flat = vgrid[:, 1, ...].reshape([-1])
        out_size = vgrid.shape[2:]
        input_transformed = _interpolate(I, x_s_flat, y_s_flat, out_size)

        output = input_transformed.reshape([B, H, W, C_img])
        return output

    output = _transform(I, vgrid)
    if train:
        output = output.permute(0, 3, 1, 2)
    return output

def get_warp(img, mesh, divide, grid_h=None, grid_w=None, start=0):

    batch_size, _, patch_size_h, patch_size_w = img.shape

    if grid_h is None:
        grid_h, grid_w = patch_size_h, patch_size_w

    src_p = get_src_p(batch_size, grid_h, grid_w, divide)
    patch_indices = get_grid(batch_size, grid_h, grid_w, 0)

    H_mat_mul = DLT_solve(src_p, mesh)

    flow, vgrid = get_flow(H_mat_mul, patch_indices, grid_h, grid_w, divide)

    grid_warp = get_grid(batch_size, grid_h, grid_w, start)[:, :2, :, :] - flow
    img_warp = transformer(img, grid_warp)
    return img_warp, flow

def get_warp_flow(img, flow, start=0):

    batch_size, _, patch_size_h, patch_size_w = flow.shape
    grid_warp = get_grid(batch_size, patch_size_h, patch_size_w, start)[:, :2, :, :] - flow
    img_warp = transformer(img, grid_warp)
    return img_warp

def upsample2d_flow_as(inputs, target_as, mode="bilinear", if_rate=False):

    _, _, h, w = target_as.size()
    if if_rate:
        _, _, h_, w_ = inputs.size()
        inputs[:, 0, :, :] *= (w / w_)
        inputs[:, 1, :, :] *= (h / h_)
    res = F.interpolate(inputs, [h, w], mode=mode, align_corners=True)
    return res

def fetch_net(params):

    if params.net_type == "basic":
        net = Net(params)
    else:
        raise NotImplementedError("Unkown model: {}".format(params.net_type))
    return net