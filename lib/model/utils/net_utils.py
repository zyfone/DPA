import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from model.utils.config import cfg
from torch.autograd import Function, Variable


def save_net(fname, net):
    import h5py

    h5f = h5py.File(fname, mode="w")
    for k, v in net.state_dict().items():
        h5f.create_dataset(k, data=v.cpu().numpy())


def load_net(fname, net):
    import h5py

    h5f = h5py.File(fname, mode="r")
    for k, v in net.state_dict().items():
        param = torch.from_numpy(np.asarray(h5f[k]))
        v.copy_(param)


def weights_normal_init(model, dev=0.01):
    if isinstance(model, list):
        for m in model:
            weights_normal_init(m, dev)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, dev)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, dev)


def clip_gradient(model, clip_norm):
    """Computes a gradient clipping coefficient based on gradient norm."""
    totalnorm = 0
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            modulenorm = p.grad.norm()
            totalnorm += modulenorm ** 2
    totalnorm = torch.sqrt(totalnorm).item()
    norm = clip_norm / max(totalnorm, clip_norm)
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            p.grad.mul_(norm)


def vis_detections(im, class_name, dets, thresh=0.8):
    """Visual debugging of detections."""
    for i in range(np.minimum(10, dets.shape[0])):
        bbox = tuple(int(np.round(x)) for x in dets[i, :4])
        score = dets[i, -1]
        if score > thresh:
            cv2.rectangle(im, bbox[0:2], bbox[2:4], (0, 204, 0), 2)
            cv2.putText(
                im,
                "%s: %.3f" % (class_name, score),
                (bbox[0], bbox[1] + 15),
                cv2.FONT_HERSHEY_PLAIN,
                1.0,
                (0, 0, 255),
                thickness=1,
            )
    return im


def adjust_learning_rate(optimizer, decay=0.1):
    """Sets the learning rate to the initial LR decayed by 0.5 every 20 epochs"""
    for param_group in optimizer.param_groups:
        param_group["lr"] = decay * param_group["lr"]


def save_checkpoint(state, filename):
    torch.save(state, filename)


def _smooth_l1_loss(
    bbox_pred,
    bbox_targets,
    bbox_inside_weights,
    bbox_outside_weights,
    sigma=1.0,
    dim=[1],
):

    sigma_2 = sigma ** 2
    box_diff = bbox_pred - bbox_targets
    in_box_diff = bbox_inside_weights * box_diff
    abs_in_box_diff = torch.abs(in_box_diff)
    smoothL1_sign = (abs_in_box_diff < 1.0 / sigma_2).detach().float()
    in_loss_box = torch.pow(in_box_diff, 2) * (sigma_2 / 2.0) * smoothL1_sign + (
        abs_in_box_diff - (0.5 / sigma_2)
    ) * (1.0 - smoothL1_sign)
    out_loss_box = bbox_outside_weights * in_loss_box
    loss_box = out_loss_box
    for i in sorted(dim, reverse=True):
        loss_box = loss_box.sum(i)
    loss_box = loss_box.mean()
    return loss_box


def _crop_pool_layer(bottom, rois, max_pool=True):
    # code modified from
    # https://github.com/ruotianluo/pytorch-faster-rcnn
    # implement it using stn
    # box to affine
    # input (x1,y1,x2,y2)
    """
    [  x2-x1             x1 + x2 - W + 1  ]
    [  -----      0      ---------------  ]
    [  W - 1                  W - 1       ]
    [                                     ]
    [           y2-y1    y1 + y2 - H + 1  ]
    [    0      -----    ---------------  ]
    [           H - 1         H - 1      ]
    """
    rois = rois.detach()
    batch_size = bottom.size(0)
    D = bottom.size(1)
    H = bottom.size(2)
    W = bottom.size(3)
    roi_per_batch = rois.size(0) / batch_size
    x1 = rois[:, 1::4] / 16.0
    y1 = rois[:, 2::4] / 16.0
    x2 = rois[:, 3::4] / 16.0
    y2 = rois[:, 4::4] / 16.0

    height = bottom.size(2)
    width = bottom.size(3)

    # affine theta
    zero = Variable(rois.data.new(rois.size(0), 1).zero_())
    theta = torch.cat(
        [
            (x2 - x1) / (width - 1),
            zero,
            (x1 + x2 - width + 1) / (width - 1),
            zero,
            (y2 - y1) / (height - 1),
            (y1 + y2 - height + 1) / (height - 1),
        ],
        1,
    ).view(-1, 2, 3)

    if max_pool:
        pre_pool_size = cfg.POOLING_SIZE * 2
        grid = F.affine_grid(
            theta, torch.Size((rois.size(0), 1, pre_pool_size, pre_pool_size))
        )
        bottom = (
            bottom.view(1, batch_size, D, H, W)
            .contiguous()
            .expand(roi_per_batch, batch_size, D, H, W)
            .contiguous()
            .view(-1, D, H, W)
        )
        crops = F.grid_sample(bottom, grid)
        crops = F.max_pool2d(crops, 2, 2)
    else:
        grid = F.affine_grid(
            theta, torch.Size((rois.size(0), 1, cfg.POOLING_SIZE, cfg.POOLING_SIZE))
        )
        bottom = (
            bottom.view(1, batch_size, D, H, W)
            .contiguous()
            .expand(roi_per_batch, batch_size, D, H, W)
            .contiguous()
            .view(-1, D, H, W)
        )
        crops = F.grid_sample(bottom, grid)

    return crops, grid


def _affine_grid_gen(rois, input_size, grid_size):

    rois = rois.detach()
    x1 = rois[:, 1::4] / 16.0
    y1 = rois[:, 2::4] / 16.0
    x2 = rois[:, 3::4] / 16.0
    y2 = rois[:, 4::4] / 16.0

    height = input_size[0]
    width = input_size[1]

    zero = Variable(rois.data.new(rois.size(0), 1).zero_())
    theta = torch.cat(
        [
            (x2 - x1) / (width - 1),
            zero,
            (x1 + x2 - width + 1) / (width - 1),
            zero,
            (y2 - y1) / (height - 1),
            (y1 + y2 - height + 1) / (height - 1),
        ],
        1,
    ).view(-1, 2, 3)

    grid = F.affine_grid(theta, torch.Size((rois.size(0), 1, grid_size, grid_size)))

    return grid


def _affine_theta(rois, input_size):

    rois = rois.detach()
    x1 = rois[:, 1::4] / 16.0
    y1 = rois[:, 2::4] / 16.0
    x2 = rois[:, 3::4] / 16.0
    y2 = rois[:, 4::4] / 16.0

    height = input_size[0]
    width = input_size[1]

    zero = Variable(rois.data.new(rois.size(0), 1).zero_())

    # theta = torch.cat([\
    #   (x2 - x1) / (width - 1),
    #   zero,
    #   (x1 + x2 - width + 1) / (width - 1),
    #   zero,
    #   (y2 - y1) / (height - 1),
    #   (y1 + y2 - height + 1) / (height - 1)], 1).view(-1, 2, 3)

    theta = torch.cat(
        [
            (y2 - y1) / (height - 1),
            zero,
            (y1 + y2 - height + 1) / (height - 1),
            zero,
            (x2 - x1) / (width - 1),
            (x1 + x2 - width + 1) / (width - 1),
        ],
        1,
    ).view(-1, 2, 3)

    return theta


class GradReverse(Function):
    def __init__(self, lambd):
        self.lambd = lambd

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        # pdb.set_trace()
        return grad_output * -self.lambd


def grad_reverse(x, lambd=1.0):
    return GradReverse(lambd)(x)


class EFocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.
            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])
        The losses are averaged across observations for each minibatch.
        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(EFocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1) * 1.0)
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        # print(N)
        C = inputs.size(1)
        # inputs = F.sigmoid(inputs)
        P = F.softmax(inputs)
        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.0)
        # print(class_mask)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P * class_mask).sum(1).view(-1, 1)
        log_p = probs.log()
        # print('probs size= {}'.format(probs.size()))
        # print(probs)
        batch_loss = -alpha * torch.exp(-self.gamma * probs) * log_p
        # print('-----bacth_loss------')
        # print(batch_loss)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.
            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])
        The losses are averaged across observations for each minibatch.
        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """

    def __init__(
        self,
        class_num,
        alpha=None,
        gamma=2,
        size_average=True,
        sigmoid=False,
        reduce=True,
    ):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1) * 1.0)
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average
        self.sigmoid = sigmoid
        self.reduce = reduce

    def forward(self, inputs, targets):
        N = inputs.size(0)
        # print(N)
        C = inputs.size(1)
        if self.sigmoid:
            P = F.sigmoid(inputs)
            # F.softmax(inputs)
            if targets == 0:
                probs = 1 - P  # (P * class_mask).sum(1).view(-1, 1)
                log_p = probs.log()
                batch_loss = -(torch.pow((1 - probs), self.gamma)) * log_p
            if targets == 1:
                probs = P  # (P * class_mask).sum(1).view(-1, 1)
                log_p = probs.log()
                batch_loss = -(torch.pow((1 - probs), self.gamma)) * log_p
        else:
            # inputs = F.sigmoid(inputs)
            P = F.softmax(inputs)

            class_mask = inputs.data.new(N, C).fill_(0)
            class_mask = Variable(class_mask)
            ids = targets.view(-1, 1)
            class_mask.scatter_(1, ids.data, 1.0)
            # print(class_mask)

            if inputs.is_cuda and not self.alpha.is_cuda:
                self.alpha = self.alpha.cuda()
            alpha = self.alpha[ids.data.view(-1)]

            probs = (P * class_mask).sum(1).view(-1, 1)

            log_p = probs.log()
            # print('probs size= {}'.format(probs.size()))
            # print(probs)

            batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
            # print('-----bacth_loss------')
            # print(batch_loss)

        if not self.reduce:
            return batch_loss
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss




class BoundaryLoss(nn.Module):

    def __init__(self):
        
        super(BoundaryLoss, self).__init__()
        self.delta = nn.Parameter(torch.randn(2).cuda())
        nn.init.normal_(self.delta)

        self.register_buffer('prototype', torch.zeros(2,128).cuda())
        self.count=0

    def forward(self, pooled_output,domain):
        pooled_output=pooled_output.detach()

        delta = F.softplus(self.delta)

        old_queue=self.prototype[domain,:].unsqueeze(0)
        c=old_queue

        d = delta[domain]
        x = pooled_output

        euc_dis = torch.norm(x - c,2, 1)
        neg_mask = (euc_dis > d).type(torch.cuda.FloatTensor)
        pos_mask = (euc_dis < d).type(torch.cuda.FloatTensor)

        neg_loss = (euc_dis - d) * neg_mask
        pos_loss = (d - euc_dis) * pos_mask
        loss = pos_loss.mean() + neg_loss.mean()

        momentum = F.cosine_similarity(pooled_output.mean(0).unsqueeze(0), old_queue)
        tmp_c=old_queue* momentum + pooled_output.mean(0).unsqueeze(0)* (1.0 - momentum)
        self.prototype[domain,:]= tmp_c

        return loss,pos_mask,neg_mask
    




from scipy.stats import norm
def global_alignment(pos_mask_s,neg_mask_s,pos_mask_t,neg_mask_t,\
    out_d_s,out_d_t,FL,step):


    neg_mask_t_index=torch.nonzero(neg_mask_t).view(-1).tolist()
    neg_mask_s_index=torch.nonzero(neg_mask_s).view(-1).tolist()
    pos_mask_s_index=torch.nonzero(pos_mask_s).view(-1).tolist()
    pos_mask_t_index=torch.nonzero(pos_mask_t).view(-1).tolist()

    if len(neg_mask_s_index)>1 and len(neg_mask_t_index)>1:
        domain_s_out = Variable(torch.zeros(len(neg_mask_s_index)).long().cuda())
        domain_t_out  = Variable(torch.ones(len(neg_mask_t_index)).long().cuda())
        out_d_s_out=out_d_s[neg_mask_s_index,:]
        out_d_t_out=out_d_t[neg_mask_t_index,:]
       
        std_s,mean_s=F.softmax(out_d_s,dim=-1)[:,0].std(0).unsqueeze(0),F.softmax(out_d_s,dim=-1)[:,0].mean(0).unsqueeze(0)
        std_t,mean_t=F.softmax(out_d_t,dim=-1)[:,1].std(0).unsqueeze(0),F.softmax(out_d_t,dim=-1)[:,1].mean(0).unsqueeze(0)

        cdf_values_s = norm.cdf(0.5, loc=mean_s.cpu().detach().numpy(), scale=std_s.cpu().detach().numpy())
        cdf_values_t = norm.cdf(0.5, loc=mean_t.cpu().detach().numpy(), scale=std_t.cpu().detach().numpy())
        cdf_values_s=cdf_values_s[0]
        cdf_values_t=1-cdf_values_t[0]
        
        dloss_s =cdf_values_s/(cdf_values_s+cdf_values_t+1e-6) *FL(out_d_s_out,domain_s_out) 
        dloss_t =cdf_values_t/(cdf_values_s+cdf_values_t+1e-6) *FL(out_d_t_out,domain_t_out)

    else:
        cdf_values_s=0
        cdf_values_t=0
        domain_s= Variable(torch.zeros(1).long().cuda())
        domain_t  = Variable(torch.ones(1).long().cuda())
        dloss_s = 0.5 *FL(out_d_s.mean(0).unsqueeze(0),domain_s) 
        dloss_t = 0.5 *FL(out_d_t.mean(0).unsqueeze(0),domain_t) 
        print("##### ERROR DA LOSS #####")


    if (step % 100== 0):
        print("domain probably")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        P_s = F.softmax(out_d_s,dim=-1)[:,0].mean(0)
        print("domain probs source: {}".format(P_s))
        print(len(neg_mask_s_index),len(pos_mask_s_index),out_d_s.shape)

        P_t = F.softmax(out_d_t,dim=-1)[:,1].mean(0)
        print("domain probs target: {}".format(P_t))
        print(len(neg_mask_t_index),len(pos_mask_t_index),out_d_t.shape)

        if len(neg_mask_s_index)>0 and len(neg_mask_t_index)>0:
            P_s_in = F.softmax(out_d_s,dim=-1)[neg_mask_s_index,0].mean(0)
            print("neg_mask_s_index center source: {}".format(P_s_in))
            P_t_in = F.softmax(out_d_t,dim=-1)[neg_mask_t_index,1].mean(0)
            print("neg_mask_t_index center target: {}".format(P_t_in))

        if len(pos_mask_s_index)>0 and len(pos_mask_t_index)>0:
            P_s_in = F.softmax(out_d_s,dim=-1)[pos_mask_s_index,0].mean(0)
            print("pos_mask_s_index center source: {}".format(P_s_in))
            P_t_in = F.softmax(out_d_t,dim=-1)[pos_mask_t_index,1].mean(0)
            print("pos_mask_t_index center target: {}".format(P_t_in))

        print("new_label_s: {} ,new_label_t: {}".format(cdf_values_s,cdf_values_t))
        print("new_label_s_weight: {} ,new_label_t_weiht: {}".format(
            cdf_values_s/(cdf_values_s+cdf_values_t+1e-6),cdf_values_t/(cdf_values_s+cdf_values_t+1e-6)))

    return dloss_s,dloss_t





def create_dict_from_list(input_list):
    # unique_values = torch.unique(torch.tensor(input_list))
    unique_values = torch.unique(input_list)

    # #remove background
    unique_values_nobg=[i for i in unique_values if i !=0]
    result_dict = {value.item(): [] for value in unique_values_nobg}
    # result_dict = {value.item(): [] for value in unique_values}

    for i, value in enumerate(input_list):
        if value.item() in unique_values_nobg:
            result_dict[value.item()].append(i)

    return result_dict


def instance_alignment_background(cls_i_s,cls_i_t,raw_ins_s,raw_ins_t,domain_ins_s,domain_ins_t):

    index_t_p=[index for index,i in enumerate(cls_i_s) if i==0]
    index_s_p=[index for index,i in enumerate(cls_i_t) if i==0]
    loss_graph=0
    loss_consis=0

    if len(index_t_p)>1 and len(index_s_p)>1:
        center_raw_s=raw_ins_s[index_s_p,:].mean(0).unsqueeze(0)
        center_domain_s=domain_ins_s[index_s_p,:].mean(0).unsqueeze(0)
        order_domain = torch.norm(domain_ins_s[index_s_p,:] - center_domain_s,2, 1)
        order_raw = torch.norm(raw_ins_s[index_s_p,:] - center_raw_s,2, 1)
        graph_s=F.cosine_similarity(order_domain.unsqueeze(0),order_raw.unsqueeze(0)).mean()
        center_raw_t=raw_ins_t[index_t_p,:].mean(0).unsqueeze(0)
        center_domain_t=domain_ins_t[index_t_p,:].mean(0).unsqueeze(0)
        order_domain_t = torch.norm(domain_ins_t[index_t_p,:] - center_domain_t,2, 1)
        order_raw_t = torch.norm(raw_ins_t[index_t_p,:] - center_raw_t,2, 1)
        graph_t=F.cosine_similarity(order_domain_t.unsqueeze(0),order_raw_t.unsqueeze(0)).mean()
        loss_graph=F.mse_loss(graph_t,graph_s.detach())
        loss_consis=loss_graph


    return loss_consis

def instance_alignment_private(cls_i_s,cls_i_t,raw_ins_s,raw_ins_t,domain_ins_s,domain_ins_t):
    
    cls_i_t_dict=create_dict_from_list(cls_i_t)
    cls_i_s_dict=create_dict_from_list(cls_i_s)
    private_cls_t=[i for i in cls_i_t_dict.keys() if i not in cls_i_s_dict.keys()]
    private_cls_t=list(set(private_cls_t))
    private_cls_s=[i for i in cls_i_s_dict.keys() if i not in cls_i_t_dict.keys()]
    private_cls_s=list(set(private_cls_s))
    index_t_p=[]
    index_s_p=[]
    for i_index in private_cls_t:
        if len(cls_i_t_dict[i_index])>1:
            index_t_p+=cls_i_t_dict[i_index]
    
    for i_index in private_cls_s:
        if len(cls_i_s_dict[i_index])>1:
            index_s_p+=cls_i_s_dict[i_index]
            
    loss_graph=0
    loss_consis=0

    if len(index_t_p)>1 and len(index_s_p)>1:
        center_raw_s=raw_ins_s[index_s_p,:].mean(0).unsqueeze(0)
        center_domain_s=domain_ins_s[index_s_p,:].mean(0).unsqueeze(0)
        order_domain = torch.norm(domain_ins_s[index_s_p,:] - center_domain_s,2, 1)
        order_raw = torch.norm(raw_ins_s[index_s_p,:] - center_raw_s,2, 1)
        graph_s=F.cosine_similarity(order_domain.unsqueeze(0),order_raw.unsqueeze(0)).mean()
        center_raw_t=raw_ins_t[index_t_p,:].mean(0).unsqueeze(0)
        center_domain_t=domain_ins_t[index_t_p,:].mean(0).unsqueeze(0)
        order_domain_t = torch.norm(domain_ins_t[index_t_p,:] - center_domain_t,2, 1)
        order_raw_t = torch.norm(raw_ins_t[index_t_p,:] - center_raw_t,2, 1)
        graph_t=F.cosine_similarity(order_domain_t.unsqueeze(0),order_raw_t.unsqueeze(0)).mean()
        loss_graph=F.mse_loss(graph_t,graph_s.detach())
        loss_consis=loss_graph

    return loss_consis