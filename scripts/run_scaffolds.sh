/home/riza/anaconda3/envs/sampling/bin/python -u runners/compute_scaffolds.py --sampling-strategy=temperature --t=1.0 --model-name=$1  --dataset-name=DRD3  &

/home/riza/anaconda3/envs/sampling/bin/python -u runners/compute_scaffolds.py --sampling-strategy=temperature --t=1.0 --model-name=$1  --dataset-name=PIN1 &

/home/riza/anaconda3/envs/sampling/bin/python -u runners/compute_scaffolds.py --sampling-strategy=temperature --t=1.0 --model-name=$1  --dataset-name=VDR &

wait