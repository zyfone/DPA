import os

net = "res101"
part = "test"
start_epoch = 8
max_epochs = 10
output_dir = "./weight_model/water2voc/result"
dataset = "water2voc"

step=[500,1000,1500]

for i in range(start_epoch, max_epochs + 1):
    model_dir = "weight_model/water2voc/water2voc_{}.pth".format(
        i
    )
    command = "CUDA_VISIBLE_DEVICES=4 python eval/test.py --cuda --gc --lc --part {} --net {} --dataset {} --model_dir {} --output_dir {} --num_epoch {}".format(
        part, net, dataset, model_dir, output_dir, i
    )
    os.system(command)
    if i>=max_epochs-1:
        for st in step:
            model_dir = "weight_model/water2voc/water2voc_{}_{}.pth".format(i,st)
            command = "CUDA_VISIBLE_DEVICES=4 python eval/test.py --cuda --gc --lc --part {} --net {} --dataset {} --model_dir {} --output_dir {} --num_epoch {}".format(
            part, net, dataset, model_dir, output_dir, i )
            os.system(command)