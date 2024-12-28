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

    def __init__(self, queue_len):
        
        super(BoundaryLoss, self).__init__()
        # self.delta = nn.Parameter(torch.randn(2).cuda())
        self.delta = nn.Parameter(torch.randn(1).cuda())
        nn.init.normal_(self.delta)

        self.hidden=queue_len
        # self.register_buffer('prototype', torch.zeros(2,self.hidden,128).cuda())
        self.register_buffer('prototype', torch.zeros(self.hidden,128).cuda())
        self.count=0

    def forward(self, pooled_output,domain):
        pooled_output=pooled_output.detach()

        delta = F.softplus(self.delta)

        # old_queue=self.prototype[domain,:,:]
        old_queue=self.prototype
        c=old_queue.mean(0).unsqueeze(0)

        d = delta#[domain]
        x = pooled_output

        euc_dis = torch.norm(x - c,2, 1)
        pos_mask = (euc_dis > d).type(torch.cuda.FloatTensor)
        neg_mask = (euc_dis < d).type(torch.cuda.FloatTensor)

        pos_loss = (euc_dis - d) * pos_mask
        neg_loss = (d - euc_dis) * neg_mask
        loss = pos_loss.mean() + neg_loss.mean()

        # momentum = F.cosine_similarity(pooled_output.mean(0).unsqueeze(0), old_queue.mean(0).unsqueeze(0))
        # tmp_c=old_queue.mean(0).unsqueeze(0) * momentum + pooled_output.mean(0).unsqueeze(0)* (1.0 - momentum)

        tmp_c=pooled_output.mean(0).unsqueeze(0)

        # if self.count<=(self.hidden-1):
        #     self.prototype[domain,self.count,:]=tmp_c
        #     self.count+=1
        # else:
        #     self.count=0           
        #     self.prototype[domain,self.count,:]=tmp_c

        
        if self.count<=(self.hidden-1):
            self.prototype[self.count,:]=tmp_c
            self.count+=1
        else:
            self.count=0           
            self.prototype[self.count,:]=tmp_c

        return loss,pos_mask,neg_mask
    




from scipy.stats import norm
def global_alignment(pos_mask_s,neg_mask_s,pos_mask_t,neg_mask_t,\
    out_d_s,out_d_t,FL,step):


    neg_mask_t_index=torch.nonzero(neg_mask_t).view(-1).tolist()
    neg_mask_s_index=torch.nonzero(neg_mask_s).view(-1).tolist()
    pos_mask_s_index=torch.nonzero(pos_mask_s).view(-1).tolist()
    pos_mask_t_index=torch.nonzero(pos_mask_t).view(-1).tolist()

    if len(pos_mask_s_index)>1 and len(pos_mask_t_index)>1:
        domain_s_out = Variable(torch.zeros(len(pos_mask_s_index)).long().cuda())
        domain_t_out  = Variable(torch.ones(len(pos_mask_t_index)).long().cuda())
        out_d_s_out=out_d_s[pos_mask_s_index,:]
        out_d_t_out=out_d_t[pos_mask_t_index,:]
        # dloss_s =0.5 *FL(out_d_s_out,domain_s_out) 
        # dloss_t =0.5 *FL(out_d_t_out,domain_t_out)

        domain_s_m = Variable(torch.zeros(1).long().cuda())
        domain_t_m  = Variable(torch.ones(1).long().cuda())
        dis_s = torch.distributions.dirichlet.Dirichlet(torch.ones(len(pos_mask_s_index))) 
        dis_t = torch.distributions.dirichlet.Dirichlet(torch.ones(len(pos_mask_t_index)))
        alpha_s = dis_s.sample((1,)).cuda()
        alpha_t = dis_t.sample((1,)).cuda()
        out_d_s_sum = torch.matmul(alpha_s, out_d_s_out)
        out_d_t_sum = torch.matmul(alpha_t, out_d_t_out)

        dloss_s =0.5 *FL(out_d_s_sum,domain_s_m) 
        dloss_t =0.5 *FL(out_d_t_sum,domain_t_m)


        std_s,mean_s=F.softmax(out_d_s,dim=-1)[:,0].std(0).unsqueeze(0),F.softmax(out_d_s,dim=-1)[:,0].mean(0).unsqueeze(0)
        std_t,mean_t=F.softmax(out_d_t,dim=-1)[:,1].std(0).unsqueeze(0),F.softmax(out_d_t,dim=-1)[:,1].mean(0).unsqueeze(0)

        cdf_values_s = norm.cdf(0.5, loc=mean_s.cpu().detach().numpy(), scale=std_s.cpu().detach().numpy())
        cdf_values_t = norm.cdf(0.5, loc=mean_t.cpu().detach().numpy(), scale=std_t.cpu().detach().numpy())
        cdf_values_s=cdf_values_s[0]
        cdf_values_t=1-cdf_values_t[0]
        
        # dloss_s =cdf_values_s *FL(out_d_s_out,domain_s_out) 
        # dloss_t =cdf_values_t *FL(out_d_t_out,domain_t_out)
        # dloss_s =cdf_values_s *FL(out_d_s_sum,domain_s_m) 
        # dloss_t =cdf_values_t *FL(out_d_t_sum,domain_t_m)
        
        
    else:
        dloss_s_m = 0 
        dloss_t_m = 0 
        cdf_values_s=0
        cdf_values_t=0
        domain_s= Variable(torch.zeros(1).long().cuda())
        domain_t  = Variable(torch.ones(1).long().cuda())
        dloss_s = 0.5 *FL(out_d_s.mean(0).unsqueeze(0),domain_s) 
        dloss_t = 0.5 *FL(out_d_t.mean(0).unsqueeze(0),domain_t) 


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
            print("in-domain center source: {}".format(P_s_in))
            P_t_in = F.softmax(out_d_t,dim=-1)[neg_mask_t_index,1].mean(0)
            print("in-domain center target: {}".format(P_t_in))

        if len(pos_mask_s_index)>0 and len(pos_mask_t_index)>0:
            P_s_in = F.softmax(out_d_s,dim=-1)[pos_mask_s_index,0].mean(0)
            print("out-domain center source: {}".format(P_s_in))
            P_t_in = F.softmax(out_d_t,dim=-1)[pos_mask_t_index,1].mean(0)
            print("out-domain center target: {}".format(P_t_in))

        print("new_label_s: {} ,new_label_t: {}".format(cdf_values_s,cdf_values_t))
        # print("dloss_s_m: {} ,dloss_t_m: {}".format(dloss_s_m.item(),dloss_t_m.item()))

    # return dloss_s+(dloss_s_m)*0.1,dloss_t+(dloss_t_m)*0.1
    return dloss_s,dloss_t