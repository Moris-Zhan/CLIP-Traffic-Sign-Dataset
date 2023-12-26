
# pip install git+https://github.com/openai/CLIP.git
# pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

import clip

print(clip.available_models())
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import pandas as pd
from PIL import Image
import torch
from glob import glob
import torchvision.transforms as trns
from torch.utils.data import random_split, DataLoader
import os
from tqdm import tqdm
from utils import *
import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
import numpy as np
torch.manual_seed(0)

"""
['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
"""

# 設置環境變量
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

current_path = "C:\\Users\\Leyan\\OneDrive\\Traffic Sign Dataset"

EPOCH = 100  # 请根据你的需求修改
# Model_name = "ViT-B/16"  # too slow
# Model_name = "ViT-L/14" # too slow
Model_name = "ViT-B/32" # 改變學習率5e-7, AdanW, 針對將18->3,  19->4, 36->30 給不同prompt text, 


Name_list = list(pd.read_csv(os.path.join(current_path, "distinct_labels.csv")).Name)

device = "cuda:0" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.
model, preprocess = clip.load(Model_name, device=device,jit=False) #Must set jit=False for training

# 輸出模型參數量
num_parameters = count_parameters(model)/1000/1000
print(f"{Model_name} 模型參數量: {num_parameters:.2f}M")
# ViT-L/14 模型參數量: 427.62M
# ViT-B/32 模型參數量: 151.28M
# ViT-B/16 模型參數量: 149.62M
# RN101 模型參數量: 119.69M
# RN50 模型參數量: 102.01M
BATCH_SIZE = 64

print(device)

if device == "cpu":
  model.float()
else :
  clip.model.convert_weights(model) # Actually this line is unnecessary since clip by default already on float16

# Create train/valid transforms
train_transform = trns.Compose([
    trns.Resize((256, 256)),
    trns.RandomCrop((224, 224)),
    trns.ColorJitter(brightness=0.2, contrast=0.2),
    # trns.RandomHorizontalFlip(), 防止誤認L變R
    trns.ToTensor(),
    trns.Normalize(mean=[0.4246, 0.4163, 0.4216], std=[0.2405, 0.2302, 0.2444]),
])

valid_transform = trns.Compose([
    trns.Resize((224, 224)),
    trns.ToTensor(),
    trns.Normalize(mean=[0.4246, 0.4163, 0.4216], std=[0.2405, 0.2302, 0.2444]),
])

init_dataset = traffic_Dataset(root=current_path + '/traffic_Data/DATA/', split='All', transform=None)

# 定义拆分的比例
lengths = [int(len(init_dataset)*0.9), int(len(init_dataset)*0.1)]
subsetA, subsetB = random_split(init_dataset, lengths)

train_dataset = MyDataset(
    subsetA, transform=train_transform
)
val_dataset = MyDataset(
    subsetB, transform=valid_transform
)

print('train data in {} split: {}'.format("train", len(train_dataset)))
print('val data in {} split: {}'.format("val", len(val_dataset)))

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


if not os.path.exists(f"{Model_name}_checkpoint/"): os.makedirs(f"{Model_name}_checkpoint/", exist_ok=True)

# 定义一些缺失的参数，例如 EPOCH 和 train_dataloader


loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), 
                        lr=5e-7, 
                        betas=(0.9,0.98), 
                        eps=1e-6, 
                        weight_decay=0.2) #Params used from paper, the lr is smaller, more safe for fine tuning to new dataset

                     

loader_step = len(train_loader)
print(f"Loader step size: {loader_step}")
# add your own code to track the training progress.

all_prompts = ["This is " + prompt for prompt in Name_list]
prev_loss = best_loss = 1000
best_epoch = 0
early_stop = 0
max_early_stop = 15
loss_values = []
acc_values = []

prev_acc = best_acc = 0

output_file_path = f'{Model_name}_history.txt'
with open(output_file_path, 'w') as f:
    sys.stdout = f  

    for epoch in range(EPOCH):
        model.train()
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()

            images, _, texts = batch

            images= images.to(device)

            texts = texts.to(device)

            logits_per_image, logits_per_text = model(images, texts.squeeze(1))

            ground_truth = torch.arange(len(images),dtype=torch.long,device=device)

            current_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2

            current_loss.backward()
            if device == "cpu":
                optimizer.step()
            else :
                convert_models_to_fp32(model)
                optimizer.step()
                clip.model.convert_weights(model)

        

        # Code to remove the old model
        if epoch > 0: os.remove(f"{Model_name}_checkpoint/model_{epoch-1}.pt")

        # Code to save the model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': current_loss,
            }, f"{Model_name}_checkpoint/model_{epoch}.pt") #just change to your preferred folder/filename

        loss_values.append(current_loss.item())


        if True:
            eval_model = model.eval()

            y_actual, y_pred =[], []

            for _, batch in tqdm(enumerate(test_loader)):

                image, lbl, texts = batch
                image= image.to(device)
                texts = texts.to(device)

                text = clip.tokenize(all_prompts).to(device)

                with torch.no_grad():
                    image_features = eval_model.encode_image(image)
                    text_features = eval_model.encode_text(text)
                    
                    logits_per_image, logits_per_text = eval_model(image, text)
                    probs = logits_per_image.softmax(dim=-1).cpu()

                    y_actual.extend(lbl.numpy())
                    y_pred.extend(torch.argmax(probs, dim=1).numpy())
                    pass


            # 比較兩個 tensor 的對應位置是否相等
            correct_predictions = torch.eq(torch.tensor(y_actual), torch.tensor(y_pred))

            # 計算精度
            accuracy = torch.sum(correct_predictions).item() / len(correct_predictions)
            acc_values.append(accuracy)

        print(f'Epoch {epoch} , Batch Size: {BATCH_SIZE}, current_loss: {current_loss.item():.2f}, Accuracy: {accuracy*100:.2f}%')

        # early stop by acc
        current_acc = accuracy*100
        if prev_acc < current_acc:
            early_stop = 0
            if best_acc < current_acc:
                if os.path.exists(f"{Model_name}_checkpoint/best.pt"): os.remove(f"{Model_name}_checkpoint/best.pt")
                best_acc = current_acc
                # early_stop = 0
                best_epoch = epoch
                print(f"Save best {Model_name} model with best_acc: {best_acc:.2f}")
                torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': current_loss,
                'acc': best_acc,
                }, f"{Model_name}_checkpoint/best.pt") #just change to your preferred folder/filename
        else:
            early_stop += 1            
            print(f"Ealry Stopping Times:{early_stop} \nCurrent {epoch}-{current_acc:.2f} Not good with {best_epoch}-{best_acc:.2f}")            
            if early_stop > max_early_stop: 
                print(f"Already stoping at Epoch {epoch}")
                break
        prev_acc = current_acc
        

    # Code to save the final model
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': current_loss,
        }, f"{Model_name}_checkpoint/final.pt") #just change to your preferred folder/filename


    # Plot training loss curve
    plt.plot(loss_values, label='Training Loss')
    plt.plot(acc_values, label='Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.title('Training Loss and Accuracy Curve')
    plt.legend()

    # Save the plotted image to a file
    plt.savefig(f'{Model_name}_checkpoint/training_loss_acc_curve.png')

    sys.stdout = sys.__stdout__

if True:
    fn = f"best.pt"
    print(fn)

    jit = True
    eval_model, _ = clip.load(Model_name, device=device,jit=jit) #Must set jit=False for training
    # print(eval_model)
    if os.path.exists(f"{Model_name}_checkpoint/{fn}"):
        checkpoint = torch.load(f"{Model_name}_checkpoint/{fn}")
        
        print(f"Load {Model_name}_checkpoint/{fn} Successed!!")
        print(f"Load Checkpoint Epoch: {checkpoint['epoch']}")
        print(f"Load Checkpoint Loss: {checkpoint['loss']}")
        epoch = checkpoint['epoch']

    checkpoint['model_state_dict']["input_resolution"] = eval_model.input_resolution
    checkpoint['model_state_dict']["context_length"] = eval_model.context_length
    checkpoint['model_state_dict']["vocab_size"] = eval_model.vocab_size
    
    eval_model.load_state_dict(checkpoint['model_state_dict'])

    print(f"Load Inference CLIP {fn} finished!")
    print(f"input_resolution: {eval_model.input_resolution.item()}")
    print(f"context_length: {eval_model.context_length.item()}")
    print(f"vocab_size: {eval_model.vocab_size.item()}")  

    y_actual, y_pred =[], []

    for _, batch in tqdm(enumerate(test_loader)):

        image, lbl, texts = batch
        image= image.to(device)
        texts = texts.to(device)

        text = clip.tokenize(all_prompts).to(device)

        with torch.no_grad():
            image_features = eval_model.encode_image(image)
            text_features = eval_model.encode_text(text)
            
            logits_per_image, logits_per_text = eval_model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu()

            y_actual.extend(lbl.numpy())
            y_pred.extend(torch.argmax(probs, dim=1).numpy())


    # 比較兩個 tensor 的對應位置是否相等
    correct_predictions = torch.eq(torch.tensor(y_actual), torch.tensor(y_pred))

    # 計算精度
    accuracy = torch.sum(correct_predictions).item() / len(correct_predictions)
    acc_values.append(accuracy)
    print(f"Best Epoch {epoch} , Accuracy: {accuracy*100:.2f}%")

if True:
    original_images = []
    images = []
    texts = []

    plt.figure(figsize=(40, 5))

    for i in range(len(Name_list)):
        for filename in [filename for filename in glob(current_path + f"/traffic_Data/DATA/{i}/*") if filename.endswith(".png") or filename.endswith(".jpg")]:

            name = os.path.basename(os.path.dirname(filename))
            image = Image.open(os.path.join(current_path, filename)).convert("RGB")

            original_images.append(image)
            images.append(preprocess(image))
            texts.append(all_prompts[int(name)])
            if(i < len(Name_list)-1): plt.subplot(3, int(len(Name_list)/3), i  + 1)  
            plt.imshow(image)
            plt.title(f"{os.path.basename(filename)}\n{all_prompts[int(name)]}")
            plt.xticks([])
            plt.yticks([])
            break

    plt.tight_layout()
    # Save the plotted image to a file
    plt.savefig(f'{Model_name}_checkpoint/all_class_sign_demo.png')

    image_input = torch.tensor(np.stack(images)).cuda()
    text_tokens = clip.tokenize(["This is " + desc for desc in texts]).cuda()

    with torch.no_grad():
        image_features = model.encode_image(image_input).float()
        text_features = model.encode_text(text_tokens).float()

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T

    count = len(Name_list)

    plt.figure(figsize=(20, 14))
    plt.imshow(similarity, vmin=0.1, vmax=0.3)
    # plt.colorbar()
    plt.yticks(range(count), texts, fontsize=5)
    plt.xticks([])
    for i, image in enumerate(original_images):
        plt.imshow(image, extent=(i - 0.5, i + 0.5, -1.6, -0.6), origin="lower")
    for x in range(similarity.shape[1]):
        for y in range(similarity.shape[0]):
            plt.text(x, y, f"{similarity[y, x]:.2f}", ha="center", va="center", size=5)

    for side in ["left", "top", "right", "bottom"]:
        plt.gca().spines[side].set_visible(False)

    plt.xlim([-0.5, count - 0.5])
    plt.ylim([count + 0.5, -2])

    plt.title("Cosine similarity between text and image features", size=20)
    # Save the plotted image to a file
    plt.savefig(f'{Model_name}_checkpoint/Cosine_similarity.png')
