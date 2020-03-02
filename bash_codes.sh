# run demo
CUDA_VISIBLE_DEVICES=1 python demo.py --net vgg16 \
               --checksession 1 --checkepoch 99 --checkpoint 505 \
               --cuda --load_dir ./models \
               --image_dir ./data/3Dircadb/3Dircadb1/3Dircadb1.1/RESAMPLE_PATIENT_PNG


ln -s /storage/lydia/PASCAL_VOC_2007/VOCdevkit VOCdevkit2007

# train vgg16 from pretrained model on ImageNet
CUDA_VISIBLE_DEVICES=0 python trainval_net.py \
                   --dataset pascal_voc --net vgg16 \
                   --bs 16 --nw 1 \
                   --lr 0.001 --lr_decay_step 25 \
                   --epochs 50 \
                   --cuda

# resume a training
CUDA_VISIBLE_DEVICES=0 python trainval_net.py \
                   --dataset pascal_voc --net vgg16 \
                   --bs 16 --nw 1 \
                   --lr 0.001 --lr_decay_step 25 \
                   --epochs 50 \
                   --r True --checksession 1 --checkepoch 25 --checkpoint 1011 \
                   --cuda


# test
CUDA_VISIBLE_DEVICES=1 python test_net.py --dataset pascal_voc --net vgg16 \
                   --checksession 1 --checkepoch 92 --checkpoint 1011 \
                   --cuda