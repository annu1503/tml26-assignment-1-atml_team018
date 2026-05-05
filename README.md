# tml26-assignment-1-atml_team018
Membership Inference Attack on a ResNet-18 image classifier – Trustworthy Machine Learning 2026

## How to Reproduce the Best Result

### 1. Clone the repository
git clone [https://github.com/YOUR_USERNAME/YOUR_REPO.git](https://github.com/annu1503/tml26-assignment-1-atml_team018.git)
cd YOUR_REPO

### 2. Install dependencies
pip install -r requirements.txt

### 3. Download the data and model
wget "https://huggingface.co/datasets/SprintML/tml26_task1/resolve/main/pub.pt" \
     "https://huggingface.co/datasets/SprintML/tml26_task1/resolve/main/priv.pt" \
     "https://huggingface.co/datasets/SprintML/tml26_task1/resolve/main/model.pt"

### 4. Run the attack
python task_template.py

This will generate `submission.csv` and automatically submit to the leaderboard.
