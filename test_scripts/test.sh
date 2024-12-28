# CUDA_VISIBLE_DEVICES=0 python eval/test.py --cuda --gc --lc \
# --part test --net res101 --dataset voc2clipart_0.75 \
# --model_dir weight_model/voc2clipart_0.75/voc2clipart_0.75_1_1000.pth \
# --output_dir ./weight_model/voc2clipart_0.75/result

python test_scripts/test_clipart_0.75.py >> test_voc075
python test_scripts/test_clipart_0.5.py >> test_voc05
python test_scripts/test_clipart_0.25.py >> test_voc025