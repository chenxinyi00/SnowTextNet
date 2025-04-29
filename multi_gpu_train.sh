# export NCCL_P2P_DISABLE=1
export NGPUS=4
CUDA_VISIBLE_DEVICES=0,1,2 python3 -m torch.distributed.launch --nproc_per_node=$NGPUS /data1/cxy/SODBNET/tools/train.py --config_file "/data1/cxy/DBNET/config/td500_resnet18_FPN_DBhead_polyLR.yaml"