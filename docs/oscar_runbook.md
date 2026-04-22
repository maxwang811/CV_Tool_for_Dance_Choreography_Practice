# Training on Oscar (Brown CCV) — runbook

This runbook covers getting the code + data onto Oscar, creating the conda
env once, submitting the training SLURM job, and pulling results back.

The training code itself is unchanged: the engine auto-uses CUDA when
available (see `src/train/engine.py`). The job just runs:

```
python -m src.train.train_pose --train configs/train/train.yaml
```

## 1. Log in and choose a working directory

```bash
ssh <your_brown_user>@ssh.ccv.brown.edu
```

Put the repo and the raw videos under `~/scratch/` (large, fast, ephemeral)
rather than `$HOME` (small quota):

```bash
mkdir -p ~/scratch/projects
cd ~/scratch/projects
git clone <your repo URL> CV_Tool_for_Dance_Choreography_Practice
cd CV_Tool_for_Dance_Choreography_Practice
```

Then upload the large inputs from your laptop. The helper script handles three
things you'd otherwise hit by hand: it pre-creates the remote directories,
multiplexes SSH so you Duo-2FA only once, and uses `--partial` so a dropped
transfer resumes on re-run.

```bash
# from the repo root on your laptop
bash scripts/push_to_oscar.sh
```

Override the defaults (user `mwang264`, host `ssh.ccv.brown.edu`, remote root
`~/scratch/projects/CV_Tool_for_Dance_Choreography_Practice`) via env vars if
needed, e.g. `OSCAR_USER=myuser bash scripts/push_to_oscar.sh`.

If you'd rather run rsync directly, first create the remote parents over SSH,
then rsync (Apple's rsync 2.6.9 does not support `--mkpath`, which is why the
bare `rsync -av --mkpath ...` form fails on macOS):

```bash
ssh <user>@ssh.ccv.brown.edu \
    "mkdir -p ~/scratch/projects/CV_Tool_for_Dance_Choreography_Practice/data/raw_videos \
              ~/scratch/projects/CV_Tool_for_Dance_Choreography_Practice/data/labels/aistpp/keypoints2d_raw"

rsync -av --progress --partial data/raw_videos/ \
    <user>@ssh.ccv.brown.edu:~/scratch/projects/CV_Tool_for_Dance_Choreography_Practice/data/raw_videos/

rsync -av --progress --partial data/labels/aistpp/keypoints2d_raw/ \
    <user>@ssh.ccv.brown.edu:~/scratch/projects/CV_Tool_for_Dance_Choreography_Practice/data/labels/aistpp/keypoints2d_raw/
```

## 2. One-time environment setup

From the repo root on an Oscar login node (or an interactive session):

```bash
bash scripts/slurm/setup_env.sh
```

This creates a conda env named `dance-cv` with Python 3.11, installs
CUDA 12.1 PyTorch wheels, then the rest of `requirements.txt`.

If `module load anaconda/2023.09-0-7nso27y` fails because CCV rotated the
module build-string, run `module avail anaconda` and re-invoke with:

```bash
ANACONDA_MODULE=anaconda/<new-build-string> bash scripts/slurm/setup_env.sh
```

The script prints `cuda available on this node: False` at the end — that
is expected on login nodes. CUDA will be available inside the GPU job.

## 3. Prepare the training JSONL splits

Also on a login node (this is CPU-only; see `README.md` for the full flags):

```bash
source activate dance-cv
python -m scripts.prepare_aist_training_data \
    --raw-videos data/raw_videos \
    --keypoints-dir data/labels/aistpp/keypoints2d_raw \
    --frames-dir data/raw_frames/aistpp \
    --out-dir data/labels/aistpp \
    --frame-stride 8
```

## 4. Submit the training job

From the repo root:

```bash
sbatch scripts/slurm/train_pose.sbatch
```

Defaults: `--partition=gpu`, `--gres=gpu:1`, `--cpus-per-task=8`,
`--mem=32G`, `--time=48:00:00`, config = `configs/train/train.yaml`.

To train a different config (e.g. HRNet) without editing the sbatch file:

```bash
sbatch --export=ALL,TRAIN_CFG=configs/train/train_hrnet.yaml \
       scripts/slurm/train_pose.sbatch
```

To get email on finish/failure, edit `scripts/slurm/train_pose.sbatch` and
set `--mail-user=your_email@brown.edu` (the `--mail-type=END,FAIL` line is
already enabled).

## 5. Monitor

```bash
squeue -u "$USER"                                         # job state
tail -f logs/slurm/cv_dance_train-<jobid>.out             # live stdout
tail -f logs/slurm/cv_dance_train-<jobid>.err             # live stderr
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS,ReqTRES   # post-run stats
scancel <jobid>                                           # cancel
```

Checkpoints are written to `data/processed/train/` (`epoch_XXXX.pt`,
`best.pt`, `last.pt`) per `configs/train/train.yaml`.

## 6. Resume from a checkpoint

The sbatch script does not resume automatically. To resume, edit
`configs/train/train.yaml` before resubmitting:

```yaml
init_from: data/processed/train/last.pt   # or epoch_0050.pt, etc.
```

The engine enforces that `init_from` paths live under `data/processed/`
(see `src/train/engine.py` and `docs/project_decisions.md` section 1).

## 7. Pull results back to your laptop

From your laptop:

```bash
rsync -av --progress \
    <user>@ssh.ccv.brown.edu:~/scratch/projects/CV_Tool_for_Dance_Choreography_Practice/data/processed/train/ \
    data/processed/train_oscar/
rsync -av --progress \
    <user>@ssh.ccv.brown.edu:~/scratch/projects/CV_Tool_for_Dance_Choreography_Practice/logs/slurm/ \
    logs/slurm_oscar/
```

## Notes and caveats

- Single GPU only. The current training engine is single-device; there is
  no DDP/multi-GPU path yet.
- `~/scratch` on Oscar is purged on a rolling basis — do not leave the
  only copy of trained weights there.
- Oscar rotates module build-strings occasionally. If a `module load`
  fails, run `module avail anaconda` and update either
  `scripts/slurm/setup_env.sh` or the `module load` line in
  `scripts/slurm/train_pose.sbatch`.
- The SBATCH script exports `OMP_NUM_THREADS`/`MKL_NUM_THREADS` to
  `$SLURM_CPUS_PER_TASK` so PyTorch DataLoader workers do not oversubscribe
  the allocated CPU cores.
