!python main.py \
--model nfnet_l0.ra2_in1k \
--batch_size 16 --epochs 200 \
--num_workers 4 \
--lr 3e-4 --lr_head 1e-3 \
--weight_decay 1e-4 \
--mlp_structures 2 \
--drop_rate 0.1