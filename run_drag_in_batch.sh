
# srun -p mm_lol -n 1 -N 1 --gres gpu:1 --job-name turbo --cpus-per-task 4 --pty \
#     python -u run_turbo_drag.py --src work_dirs/2023-12-06_21:13  \
#         --save-dir work_dirs/l3-lr005-lam005 \
#         --lr 0.05 --lam 0.05 \
#         --layer-idx 3

# srun -p mm_lol -n 1 -N 1 --gres gpu:1 --job-name turbo --cpus-per-task 4 --pty \
#     python -u run_turbo_drag.py --src work_dirs/2023-12-06_21:13  \
#         --save-dir work_dirs/l2-lr005-lam005 \
#         --lr 0.05 --lam 0.05 \
#         --layer-idx 2

srun -p mm_lol -n 1 -N 1 --gres gpu:1 --job-name turbo --cpus-per-task 4 --pty \
    python -u run_turbo_drag.py --src work_dirs/2023-12-06_21:13  \
        --save-dir work_dirs/l23-lr005-lam002 \
        --lr 0.05 --lam 0.02 \
        --layer-idx 2 3
