python -u runners/compute_scaffolds.py --sampling-strategy=temperature --t=1.0 --model-name=$1  --dataset-name=DRD3  &

python -u runners/compute_scaffolds.py --sampling-strategy=temperature --t=1.0 --model-name=$1  --dataset-name=PIN1 &

python -u runners/compute_scaffolds.py --sampling-strategy=temperature --t=1.0 --model-name=$1  --dataset-name=VDR &

wait