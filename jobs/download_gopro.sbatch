#!/bin/bash
#SBATCH --job-name=gopro-dl
#SBATCH --partition=biggpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/home-mscluster/kraman/FVR_Project/gopro-dl-%j.log

FVR=/home-mscluster/kraman/.conda/envs/fvr
mkdir -p ~/FVR_Project/data/raw/gopro
cd ~/FVR_Project/data/raw/gopro

echo "=== attempt 1: gdown --fuzzy with full URL ==="
$FVR/bin/gdown "https://drive.google.com/uc?id=1y4wvPdOG3mojpFCHTqLgriexhbjoWVkK" -O GOPRO_Large.zip

# if attempt 1 produced nothing usable, try the folder-style URL form
if [ ! -s GOPRO_Large.zip ]; then
    echo "=== attempt 2: gdown --fuzzy with /file/d/ URL ==="
    $FVR/bin/gdown "https://drive.google.com/file/d/1y4wvPdOG3mojpFCHTqLgriexhbjoWVkK/view" -O GOPRO_Large.zip
fi

echo "=== download result ==="
ls -lah ~/FVR_Project/data/raw/gopro/

# only unzip if we actually got a real (non-tiny) file
if [ -s GOPRO_Large.zip ] && [ "$(stat -c%s GOPRO_Large.zip)" -gt 1000000 ]; then
    echo "=== unzipping ==="
    unzip -q -o GOPRO_Large.zip -d ~/FVR_Project/data/raw/gopro/
    echo "=== structure (first 3 levels) ==="
    find ~/FVR_Project/data/raw/gopro -maxdepth 3 -type d | head -30
else
    echo "!! download failed or file too small -- check for quota/permission errors above"
fi
echo "=== done ==="
