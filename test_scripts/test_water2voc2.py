import os

net = "res101"
part = "test"
start_epoch = 9
max_epochs = 9
output_dir = "./weight_model/water2voc/result"
dataset = "water2voc"

for i in range(start_epoch, max_epochs + 1):
    model_dir = "weight_model/water2voc/water2voc_{}.pth".format(
        i
    )
    command = "CUDA_VISIBLE_DEVICES=8 python eval/test.py --cuda --gc --lc --part {} --net {} --dataset {} --model_dir {} --output_dir {} --num_epoch {}".format(
        part, net, dataset, model_dir, output_dir, i
    )
    os.system(command)