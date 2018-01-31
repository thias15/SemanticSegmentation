#python test_enet.py --dataset_name=RSSRC2c --combine_val=True --save_step=1 --network="ErfNet_Small" --num_classes=2 --checkpoint_dir=./log/train_RSSCityscapes2c_ErfNet_Small_ENET_lr_0.001_bs_10 --logdir=./log/test_RSSCityscapes2c_ErfNet_Small_ENET_lr_0.001_bs_10

#RSSCarla2c
#RSSRC2c

#Cityscapes
python test_enet.py --dataset_name=RSSCarla2c --combine_val=True --network="ErfNet_Small" --num_classes=2 --checkpoint_dir=./log/train_RSSCityscapes2c_ErfNet_Small_ENET_lr_0.001_bs_10 --logdir=./log/test_RSSCityscapes2c_ErfNet_Small_ENET_lr_0.001_bs_10

#Cityscapes - Aug
python test_enet.py --dataset_name=RSSRC2c --combine_val=True --network="ErfNet_Small" --num_classes=2 --checkpoint_dir=./log/train_RSSCityscapes2c_ErfNet_Small_ENET_lr_0.001_bs_10_aug_True --logdir=./log/test_RSSCityscapes2c_ErfNet_Small_ENET_lr_0.001_bs_10_aug_True

#Berkeley
python test_enet.py --dataset_name=RSSRC2c --combine_val=True --network="ErfNet_Small" --num_classes=2 --checkpoint_dir=./log/train_RSSBerkeley2c_ErfNet_Small_ENET_lr_0.001_bs_10 --logdir=./log/test_RSSBerkeley2c_ErfNet_Small_ENET_lr_0.001_bs_10

#Carla
python test_enet.py --dataset_name=RSSRC2c --combine_val=True --network="ErfNet_Small" --num_classes=2 --checkpoint_dir=./log/train_RSSCarla2c_ErfNet_Small_ENET_lr_0.001_bs_10 --logdir=./log/test_RSSCarla2c_ErfNet_Small_ENET_lr_0.001_bs_10

#CamVid
python test_enet.py --dataset_name=RSSRC2c --combine_val=True --network="ErfNet_Small" --num_classes=2 --checkpoint_dir=./log/train_RSSCamVid2c_ErfNet_Small_ENET_lr_0.001_bs_10 --logdir=./log/test_RSSCamVid2c_ErfNet_Small_ENET_lr_0.001_bs_10

#RC
python test_enet.py --dataset_name=RSSRC2c --combine_val=True --network="ErfNet_Small" --num_classes=2 --checkpoint_dir=./log/train_RSSRC2c_ErfNet_Small_ENET_lr_0.001_bs_10 --logdir=./log/test_RSSRC2c_ErfNet_Small_ENET_lr_0.001_bs_10

#All
python test_enet.py --dataset_name=RSSRC2c --combine_val=True --network="ErfNet_Small" --num_classes=2 --checkpoint_dir=./log/train_RSSAll2c_ErfNet_Small_ENET_lr_0.001_bs_10 --logdir=./log/test_RSSAll2c_ErfNet_Small_ENET_lr_0.001_bs_10



#Cityscapes - Regular ErfNet
python test_enet.py --dataset_name=RSSCarla2c --combine_val=True --network="ErfNet" --num_classes=2 --checkpoint_dir=./log/train_RSSCityscapes2c_ErfNet_ENET_lr_0.001_bs_10_aug_False --logdir=./log/test_RSSCityscapes2c_ErfNet_ENET_lr_0.001_bs_10_aug_False

