# CLIP-Traffic-Sign-Dataset
![image](https://github.com/Moris-Zhan/CLIP-Traffic-Sign-Dataset/assets/24097516/9b5e8599-a320-4e17-8195-e19974b0a8b0)

Contrastive Fine-Tuning on Traffic Sign Dataset

# Desktop
OS	Windows 10
CPU	I7-12700 (2100Mhz) 12-core
Ram	64G
GPU	NVIDIA GeForce RTX 3060 (12G) * 1
Python	3.8.18
PyTorch	1.7.1+cu110
NumPy	1.24.3

# Fine-Tune
LR		5e-6
betas		(0.9,0.98)
eps		5e-6
weight_decay		0.2
EPOCH		100
BATCH_SIZE		64
optimizer		Adam
Early Stop		15

# Data Agumation		
• Train(90%)    
  ◦ Resize	(256, 256)    
  ◦ RandomCrop	(224, 224)    
  ◦ Normaliztion	 mean=[0.4246, 0.4163, 0.4216], std=[0.2405, 0.2302, 0.2444]
			
• Valid(10%)    
  ◦ Normaliztion	mean=[0.4246, 0.4163, 0.4216], std=[0.2405, 0.2302, 0.2444]

# Insight
Total number of classes: 58
Total number of samples: 4170
Traffic sign can’t use RandomHorizontalFlip
![image](https://github.com/Moris-Zhan/CLIP-Traffic-Sign-Dataset/assets/24097516/d435d725-0256-4c45-83a2-4942ff8229ed)

* Top3 most
watch out for cars      446
No stopping             324
Speed limit (40km/h)    268
![image](https://github.com/Moris-Zhan/CLIP-Traffic-Sign-Dataset/assets/24097516/7062428e-3ebc-463f-9280-ae045bb1411d)

* Top3 less
keep Left                    2
Dont Go straight or Right    2
Give Way                     2
Traffic signals              4
Unknown8                     6
![image](https://github.com/Moris-Zhan/CLIP-Traffic-Sign-Dataset/assets/24097516/3c7e6575-3d77-44b5-b3a0-1790b8fbe0a3)

# Un-unique classes
Bicycles crossing    2
Speed limit (40km/h) 2
Speed limit (50km/h) 2

# Un-unique classes
3  Speed limit (40km/h, Red)
4  Speed limit (50km/h, Red)
18 Speed limit (40km/h, Black)
19 Speed limit (50km/h, Black)
30 Bicycles crossing, Circle
36 Bicycles crossing, Triangle

# Fine-Tune Result
| Model                               | Best Train Loss | Best Accuracy | Best Epoch |
|-------------------------------------|-----------------|---------------|------------|
| ViT-B/32 – baseline (lr=5e-6)        | 0.54            | 96.16%        | 72         |
| ViT-B/32 – Exp1 (lr=5e-7)            | 0.65            | 97.6%         | 93         |
| ViT-B/32 – Exp2 (mod Exp1)           | 0.96            | 97.36%        | 82         |
| ViT-B/32 – Exp3 (fine-tune Exp2)     | 0.77            | 98.32%        | 34         |

* Exp1 (early stop By Acc)
* Exp2 (mod Exp1)(distinct prompt text) (change brightness, contrast)
* Exp3 (fine-tune Exp2) (lr=5e-9, weight_decay=0.2, eta_min=5e-12) 

# Class Cosine similarity
![image](https://github.com/Moris-Zhan/CLIP-Traffic-Sign-Dataset/assets/24097516/04f3e8ce-e084-4a1d-a2b0-d5bbd2f55a2a)

# Inference - Auto Classify
![image](https://github.com/Moris-Zhan/CLIP-Traffic-Sign-Dataset/assets/24097516/47040b6b-111d-44bb-bc1a-06bd3aa851c3)

# Inference - Ask Traffic Sign
![image](https://github.com/Moris-Zhan/CLIP-Traffic-Sign-Dataset/assets/24097516/47f909cd-8ba7-4067-8db0-5cad43205fe3)






