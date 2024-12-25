dataset_names=("DRD3" "PIN1" "VDR")

for dataset_name in ${dataset_names[@]}; do
    temperatures=(0.5 0.75 1.25)
    for t in ${temperatures[@]}; do
        /home/riza/anaconda3/envs/sampling/bin/python -u runners/compute_descriptors.py --sampling-strategy=temperature --t=$t --model-name=$1  --dataset-name=$dataset_name  &
    done
    wait

    temperatures=(1.5 1.75 2.0)
    for t in ${temperatures[@]}; do
        /home/riza/anaconda3/envs/sampling/bin/python -u runners/compute_descriptors.py --sampling-strategy=temperature --t=$t --model-name=$1  --dataset-name=$dataset_name  &
    done
    wait

    topks=(3 5 10 15)
    for topk in ${topks[@]}; do
        /home/riza/anaconda3/envs/sampling/bin/python -u runners/compute_descriptors.py --sampling-strategy=topk --t=1.0 --model-name=$1  --dataset-name=$dataset_name --k=$topk  &
    done
    
    topks=(20 25 30)
    for topk in ${topks[@]}; do
        /home/riza/anaconda3/envs/sampling/bin/python -u runners/compute_descriptors.py --sampling-strategy=topk --t=1.0 --model-name=$1  --dataset-name=$dataset_name --k=$topk  &
    done
    wait

    topps=(0.5 0.6 0.7 0.8)
    for topp in ${topps[@]}; do
        /home/riza/anaconda3/envs/sampling/bin/python -u runners/compute_descriptors.py --sampling-strategy=topp --t=1.0 --model-name=$1  --dataset-name=$dataset_name --p=$topp  &
    done
    wait
    
    topps=(0.9 0.925 0.950 0.975)
    for topp in ${topps[@]}; do
        /home/riza/anaconda3/envs/sampling/bin/python -u runners/compute_descriptors.py --sampling-strategy=topp --t=1.0 --model-name=$1  --dataset-name=$dataset_name --p=$topp  &
    done
    wait

done
wait
