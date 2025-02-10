import os

net = "res101"
part = "test"
start_epoch = 6
max_epochs = 10
output_dir = "./weight_model/voc2clipart_0.75/result"
dataset = "voc2clipart_0.75"

# step=[1000,2000,3000,4000,5000,6000,7000,8000,9000]
step=[500,1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500]

# step=[ 7500, 8000, 8500, 9000, 9500]




for i in range(start_epoch, max_epochs + 1):
    model_dir = "weight_model/voc2clipart_0.75/voc2clipart_0.75_{}.pth".format(
        i
    )
    command = "CUDA_VISIBLE_DEVICES=0 python eval/test.py --cuda --gc --lc --part {} --net {} --dataset {} --model_dir {} --output_dir {} --num_epoch {}".format(
        part, net, dataset, model_dir, output_dir, i
    )
    os.system(command)
    if i>=max_epochs-1:
        for st in step:
            model_dir = "weight_model/voc2clipart_0.75/voc2clipart_0.75_{}_{}.pth".format(i,st)
            command = "CUDA_VISIBLE_DEVICES=0 python eval/test.py --cuda --gc --lc --part {} --net {} --dataset {} --model_dir {} --output_dir {} --num_epoch {}".format(
            part, net, dataset, model_dir, output_dir, i )
            os.system(command)

