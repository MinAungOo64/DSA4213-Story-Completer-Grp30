# DSA4213-Generator-Grp30
This folder contains the following:
- The final model within full_model_best
- Masked datasets
- Generator Data Augmentation.ipynb
- Generator Inference.ipynb
- Generator Training.ipynb

#### The "Generator Data Augmentation.ipynb" file is used to generate the masked data
#### The "Generator Training & Inference.ipynb" file is used to train and obtain the best model
#### The "Generator Inference.ipynb" file is used to test the best model on inference

## The following are packages required to run the Generator files

### For the "Generator Data Augmentation.ipynb" file:
```shell
import json
import re
from datasets import Dataset
from datasets import load_from_disk
from collections import Counter
import random
```

### For the "Generator Training.ipynb" file:
```shell
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_scheduler
from datasets import load_from_disk  # or load_dataset if remote
import re
from tqdm import tqdm
```

### For the "Generator Inference.ipynb" file:
```shell
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_from_disk
```
