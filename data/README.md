# Dataset

The notebook uses the **training** folder of [*You Snooze You Win: The PhysioNet/CinC Challenge 2018*](https://physionet.org/content/challenge-2018/1.0.0/), which is publicly available on PhysioNet and in the associated [Google Cloud Storage bucket](https://console.cloud.google.com/storage/browser/challenge-2018-1.0.0.physionet.org). It holds 994 labeled recordings (about 135 GB) and must end up in `data/training/<record>/`, e.g. `data/training/tr03-0005/tr03-0005.mat`.

## Download

Use one of the two scripts in this folder. Each asks which folders to fetch and always saves them next to itself, i.e. in `data/`, regardless of where you call it from:

| Script | Source | Requirements |
|--------|--------|--------------|
| `download_web.sh` | PhysioNet website | `wget`; runs one background job per record prefix (`tr03`…`tr14`), with progress in `data/wget-log*` |
| `download_cloud.sh` | Google Cloud bucket (usually faster) | [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) (`gsutil`) |

```bash
bash data/download_web.sh     # or: bash data/download_cloud.sh
```

If your disk space is elsewhere, download there and symlink the folder: `ln -s /big/disk/training data/training`.

Before training, make sure the download has finished. The notebook checks that every record in the split is present and stops with a list of missing records otherwise.

## Training / validation / test split

The labeled `training` folder is split into training (60%), validation (15%) and test (25%) sets. The split used in the paper is stored in each `models/<model>/records.csv` and is reused automatically. A new model folder without `records.csv` gets a fresh, seeded split.

## The PhysioNet `test` folder

The PhysioNet `test` folder (989 recordings) contains **no arousal labels**, because they are not publicly available. It can't be used for training or scoring, so the notebook ignores it. The download scripts still offer it, but you don't need it.

## Generated files

On first run the notebook writes `RECORDS.txt` manifests and caches each preprocessed record next to its source as `<record>.h5py` (or `<record>_norm.h5py` when Z-normalized) plus `<record>-arousal.h5py`. These files are git-ignored. Delete them to force preprocessing again.
