import torch
import numpy as np




def weight_compute(instance_sigmoid,same_size_label,gmm_split):

    with torch.no_grad():

        instance_sigmoid=instance_sigmoid.detach()
        same_size_label=same_size_label.detach()
        weight_ins=torch.ones(instance_sigmoid.size(0)).cuda()
        
        g =  torch.abs(instance_sigmoid.squeeze(-1) - same_size_label.squeeze(-1)).detach()
        std=g.std(0)
        mean=g.mean(0)        

        weight_center=(1.0-torch.abs(mean-0.5)/0.5)

        split=min((g.max()-g.min())*std,gmm_split)

        edges  = torch.arange(g.min().item(), g.max().item(),split).cuda()
        edges_max=g.max()+1e-2
        edges_min=g.min()-1e-2
        edges=torch.cat((edges_min.unsqueeze(0),edges,edges_max.unsqueeze(0)))
        ignore_index=g_split_2(g,edges,1)
        ignore_index_all=ignore_index
        weight_ins*=weight_center
        weight_ins[ignore_index_all]=0
       
    return weight_ins


def get_sublist_between_elements(lst, start_element, end_element):
    try:
        start_index = lst.index(start_element)
        end_index = lst.index(end_element)
        
        if start_index <= end_index:
            return lst[start_index:end_index + 1]
        else:
            return []
    except ValueError:
        return []



def g_split_2(g,edges,len_val):
    g=g.detach().cpu().numpy()
    edges=edges.detach().cpu().numpy()
    bins_index_list={} #id bins bags
    num_list={} #frequency

    for i in range(edges.size-1):
        inds = (g >= edges[i]) & (g < edges[i + 1]) 

        index_d= np.nonzero(inds)[0].tolist()
        num_in_bin = inds.sum()

        bins_index_list[i]=index_d
        num_list[i]=num_in_bin

            
    index_list = [i for i in range(len(bins_index_list))  if len(bins_index_list[i]) > 0]
    in_list= find2(index_list,split_len=len_val)
    out_list=[i for i in index_list if i not in in_list]  
 
    begin_out_list=[i for i in out_list if i <in_list[0]]
    end_out_list=[i for i in out_list if i >in_list[-1]]
    min_begin=in_list[0]
    max_end=in_list[-1]
    max_frequency_begin=num_list[in_list[0]]
    max_frequency_end=num_list[in_list[-1]]
    if  len(begin_out_list)>0 and num_list[in_list[0]]>1: 
        for i in begin_out_list[::-1]:
            if num_list[i]>=max_frequency_begin:
                min_begin=i
                max_frequency_begin=num_list[i]
    if  len(end_out_list)>0 and num_list[in_list[-1]]>1:
        for i in end_out_list:
            if num_list[i]>=max_frequency_end:
                max_end=i
                max_frequency_end=num_list[i]

    in_list=get_sublist_between_elements(index_list,min_begin,max_end)
    out_list=[i for i in index_list if i not in in_list]   
    out_list=remove_consecutive(out_list)# remove noise
    ignore_index=[]
    all=0
    gaussian_index=[]
    for i in index_list:
        if i in out_list:
            ignore_index+=bins_index_list[i] 
        elif i in in_list:
            gaussian_index+=bins_index_list[i]
        all+=num_list[i]
    
    return ignore_index 


def find2(arr,split_len):
    if not arr:
        return []
    longest_sequence = [arr[0]] 
    current_sequence = [arr[0]]  

    for i in range(1, len(arr)):
        if arr[i] - arr[i - 1] > split_len:
            current_sequence = [arr[i]]
        else:
            current_sequence.append(arr[i])

        if len(current_sequence) > len(longest_sequence):
            longest_sequence = current_sequence

    return longest_sequence
   

def remove_consecutive(lst):
    result = []
    i = 0
    while i < len(lst):
        start = lst[i]
        end = start
        while i < len(lst) - 1 and lst[i + 1] == end + 1:
            i += 1
            end = lst[i]
        if start == end:
            result.append(start)
        i += 1
    return result