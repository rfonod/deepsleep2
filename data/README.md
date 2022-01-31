# Dataset Availability

This folder shall contain the **training** and/or **test** folder(s) of the *You Snooze You Win - The PhysioNet Computing in Cardiology Challenge 2018*. The folders are publicly available on the challenge's [Official Web Page](https://physionet.org/content/challenge-2018/1.0.0/) or associated [Google Cloud Storage](https://console.cloud.google.com/storage/browser/challenge-2018-1.0.0.physionet.org). Each folder is approximately 135 GB in size. 

The content of these folders can be downloaded using the provided scripts in this folder:
- `download_web.sh` : to download the **training** and/or **test** folder(s) from the web
- `download_cloud.sh` : to download the **training** and/or **test** folder(s) from the cloud

**_Note:_** If only the **training** folder is available, then the data in this folder will be split into training, validation, and test sets. If both the **training** and **test** folders are available, then the data in the **training** folder will be split into training and validation sets, while the data in the **test** folder will be used for testing.