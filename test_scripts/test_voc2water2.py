import os

net = "res101"
part = "test"
start_epoch = 5
max_epochs = 5
output_dir = "./weight_model/voc2water/result"
dataset = "voc2water"

# step=[ 7500, 8000, 8500, 9000, 9500]


for i in range(start_epoch, max_epochs + 1):
    model_dir = "weight_model/voc2water/voc2water_{}.pth".format(
        i
    )
    command = "CUDA_VISIBLE_DEVICES=6 python eval/test.py --cuda --gc --lc --part {} --net {} --dataset {} --model_dir {} --output_dir {} --num_epoch {}".format(
        part, net, dataset, model_dir, output_dir, i
    )
    os.system(command)
    # if i>=max_epochs-1:
    #     for st in step:
    #         model_dir = "weight_model/voc2water/voc2water_{}_{}.pth".format(i,st)
    #         command = "CUDA_VISIBLE_DEVICES=0 python eval/test.py --cuda --gc --lc --part {} --net {} --dataset {} --model_dir {} --output_dir {} --num_epoch {}".format(
    #         part, net, dataset, model_dir, output_dir, i )
    #         os.system(command)
