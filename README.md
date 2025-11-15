# DSA4213-Story-Completer-Grp30



## Installation guide
### 1. Clone the repository to your computer  
Run this in your terminal: 

```shell
cd Desktop
git clone https://github.com/MinAungOo64/DSA4213-Story-Completer-Grp30.git
```

Then head inside:
```shell
cd DSA4213-Story-Completer-Grp30
```
Or you can just open it manually.

### 2. Create and activate virtual environment
```shell
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install PyTorch with CUDA (GPU support)
Before installing the rest of the dependencies, install PyTorch according to your CUDA version.

Visit the official installation page below and follow the instructions:

> https://pytorch.org/get-started/locally/

Example command for CUDA 12.6:
```shell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

If you do not have a CUDA-capable GPU, install the CPU version instead:
```shell
pip install torch torchvision torchaudio
```

### 4. Install remaining dependencies
Once PyTorch is installed, run:
```shell
pip install -r requirements.txt
```

### 5. Main entry point to reproduce final results
After installation, you can execute the main script with:
```shell
python _____.py
```
