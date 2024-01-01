!python main.py \
--model convnext_small.fb_in22k \
--batch_size 16 --epochs 200 \
--num_workers 4 \
--lr 3e-4 --lr_head 1e-3 \
--weight_decay 1e-5 \
--mlp_structures 2 \
--drop_rate 0.1