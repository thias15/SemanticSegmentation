python train_enet.py --dataset_name="Berkeley" --validation_name="Berkeley" --weighting="ENET" --network="ENet_Small" --initial_learning_rate=1e-3 --decay_steps=50000 --num_epochs=1000

python train_enet.py --dataset_name="Cityscapes" --validation_name="Cityscapes" --weighting="ENET" --network="ENet_Small" --initial_learning_rate=1e-3 --decay_steps=50000 --num_epochs=1000 

python train_enet.py --dataset_name="Berkeley" --validation_name="Berkeley" --weighting="ENET" --network="ErfNet_Small" --initial_learning_rate=1e-3 --decay_steps=50000 --num_epochs=1000
python train_enet.py --dataset_name="Cityscapes" --validation_name="Cityscapes" --weighting="ENET" --network="ErfNet_Small" --initial_learning_rate=1e-3 --decay_steps=50000 --num_epochs=1000

python train_enet.py --dataset_name="Berkeley" --validation_name="Berkeley" --weighting="ENET" --network="ErfNet_NoDS" --initial_learning_rate=1e-3 --decay_steps=50000 --num_epochs=1000
python train_enet.py --dataset_name="Cityscapes" --validation_name="Cityscapes" --weighting="ENET" --network="ErfNet_NoDS" --initial_learning_rate=1e-3 --decay_steps=50000 --num_epochs=1000


python train_enet.py --dataset_name="CVPR02Noise" --validation_name="CVPRVal" --weighting="ENET" --network="ENet_Small" --num_epochs=100

#python train_enet.py --dataset_dir="./dataset/CamVid" --weighting="MFB" --num_epochs=500 --logdir="./log/train_MFB_combined_data" --combine_dataset=True
#python train_enet.py --dataset_dir="./dataset/CamVid" --weighting="ENET" --num_epochs=500 --logdir="./log/train_ENet_combined_data" --combine_dataset=True

#python train_enet.py --dataset_dir="./dataset/CamVid" --weighting="MFB" --num_epochs=500 --logdir="./log/train_MFB"
#python train_enet.py --dataset_dir="./dataset/CamVid" --weighting="ENET" --num_epochs=500 --logdir="./log/train_ENet"
