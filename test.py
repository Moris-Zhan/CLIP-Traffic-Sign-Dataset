import torch
import clip
from PIL import Image
from glob import glob
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import random
import re

# 設置環境變量
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

fn = f"best.pt"
jit = True

# Model_name = "ViT-B/16"  # too slow
Model_name = "ViT-B/32"
# Model_name = "ViT-L/14"
device = "cuda:0" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.
current_path = "C:\\Users\\Leyan\\OneDrive\\Traffic Sign Dataset"

# Name_list = list(pd.read_csv(os.path.join(current_path, "labels.csv")).Name)
Name_list = list(pd.read_csv(os.path.join(current_path, "distinct_labels.csv")).Name)
text_descriptions = [f"This is {prompt}" for prompt in Name_list]

eval_model, preprocess = clip.load(Model_name, device=device,jit=jit) #Must set jit=False for training
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
else:
    print("Load failed")


# Cosine_similarity + Top5-Prob (All Train Label)
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
            texts.append(text_descriptions[int(name)])
            if(i < len(Name_list)-1): plt.subplot(3, int(len(Name_list)/3), i  + 1)  
            plt.imshow(image)
            plt.title(f"{os.path.basename(filename)}\n{text_descriptions[int(name)]}")
            plt.xticks([])
            plt.yticks([])
            break

    plt.tight_layout()
    # Save the plotted image to a file
    plt.savefig(f'{Model_name}_checkpoint/all_class_sign_demo.png')

    image_input = torch.tensor(np.stack(images)).cuda()
    text_tokens = clip.tokenize(["This is " + desc for desc in texts]).cuda()

    with torch.no_grad():
        image_features = eval_model.encode_image(image_input).float()
        text_features = eval_model.encode_text(text_tokens).float()

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

    plt.figure(figsize=(32, 128))

    with torch.no_grad():
        text_features = eval_model.encode_text(text_tokens).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    top_probs, top_labels = text_probs.cpu().topk(5, dim=-1)

    for i, image in enumerate(original_images):
        plt.subplot(32, 4, 2 * i + 1)
        plt.imshow(image)
        plt.axis("off")

        plt.subplot(32, 4, 2 * i + 2)
        y = np.arange(top_probs.shape[-1])
        plt.grid()
        plt.barh(y, top_probs[i])
        plt.gca().invert_yaxis()
        plt.gca().set_axisbelow(True)
        plt.yticks(y, [Name_list[index] for index in top_labels[i].numpy()])
        plt.xlabel("probability")

    plt.subplots_adjust(wspace=0.5)
    plt.savefig(f'{Model_name}_checkpoint/all_class_top5.png')

# 詢問特定圖片類別Top5-Prob
if True:
    os.makedirs(f'{Model_name}_checkpoint/random/', exist_ok=True)
    c = 15
    while(c>0):
        c -= 1
        original_images = []
        images = []
        texts = []
        image_list = glob(current_path + f"/traffic_Data/TEST/*")

        filename = random.sample(image_list, 1)[0]
        image = Image.open(os.path.join(current_path, filename)).convert("RGB")

        original_images.append(image)
        images.append(preprocess(image))

        image_input = torch.tensor(np.stack(images)).cuda()
        text_descriptions = [f"This is {label}" for label in Name_list]
        text_tokens = clip.tokenize(text_descriptions).cuda()

        with torch.no_grad():
            image_features = eval_model.encode_image(image_input).float()
            text_features = eval_model.encode_text(text_tokens).float()
            text_features /= text_features.norm(dim=-1, keepdim=True)

        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        top_probs, top_labels = text_probs.cpu().topk(5, dim=-1)

        plt.figure(figsize=(15, 10))

        for i, image in enumerate(original_images):
            print(f"Image:{os.path.basename(filename)}, the most likely class {Name_list[top_labels[0][0].item()]}")
            plt.subplot(1, 2, 1)
            plt.imshow(image)
            plt.axis("off")

            plt.subplot(1, 2, 2)
            y = np.arange(top_probs.shape[-1])
            plt.grid()
            plt.barh(y, top_probs[i])
            plt.gca().invert_yaxis()
            plt.gca().set_axisbelow(True)
            plt.yticks(y, [Name_list[index] for index in top_labels[i].numpy()])
            plt.xlabel(f"Image:{os.path.basename(filename)}, probability")

        plt.subplots_adjust(wspace=0.5)
        plt.savefig(f'{Model_name}_checkpoint/random/{"".join(os.path.basename(filename).split(".")[:-1])}_Top5-Prob.png')

# 詢問特定交通號誌類別(I want to find ...)
if True:
    os.makedirs(f'{Model_name}_checkpoint/random/', exist_ok=True)
    a = 15
    # count = 0
    while(a>0):
        a -= 1
        original_images = []
        images = []
        path_list = []
        Prob_thres = 0.45

        # key = random.sample(Name_list, 1)[0]
        key = Name_list[a]
        a += 1
        # key = "Bicycles crossing"
        print(f"Prob_thres > {Prob_thres}")
        print(f"Search Key: {key}")


        plt.figure(figsize=(20, 5))

        image_list = glob(current_path + f"/traffic_Data/TEST/*")[:]
        # image_list = random.sample(image_list, 50)

        for i, filename in enumerate([filename for filename in image_list if filename.endswith(".png") or filename.endswith(".jpg")]):
            image = Image.open(os.path.join(current_path, filename)).convert("RGB")
            path_list.append(os.path.basename(filename))
        
            # plt.subplot(5, int(len(image_list)/5), i  + 1)  
            original_images.append(image)
            images.append(preprocess(image))

            # if(i < len(image_list)-1): plt.subplot(5, int(len(image_list)/5), i  + 1) 

            # plt.imshow(image)
            # plt.title(f"{os.path.basename(filename)}\n")
            # plt.xticks([])
            # plt.yticks([])

        # plt.tight_layout()
        # plt.savefig(f'{Model_name}_checkpoint/random/sign_list.png')

        image_input = torch.tensor(np.stack(images)).cuda()
        text_descriptions = [f"This is {key}"]
        text_tokens = clip.tokenize(text_descriptions).cuda()

        with torch.no_grad():
            image_features = eval_model.encode_image(image_input).float()  
            text_features = eval_model.encode_text(text_tokens).float()
            text_features /= text_features.norm(dim=-1, keepdim=True)

        key_probs = (100.0 * text_features @ image_features.T).softmax(dim=-1)[0]
        nonzero_indices = torch.nonzero(key_probs > Prob_thres).squeeze()
        print(key_probs[key_probs > 0.2])

        count = len(nonzero_indices)
        print(f"Find Probably {key}, Img num: {count}")
        if count > 0:
            plt.figure(figsize=(5, 5))
            if nonzero_indices.dim() == 0: nonzero_indices = nonzero_indices.unsqueeze(0)
            for idx, indice in enumerate(nonzero_indices):
                plt.subplot(1, count, idx  + 1)  
                plt.imshow(original_images[indice])
                plt.title(f"{path_list[indice]} \nProb:{key_probs[indice].item():.2f}")

                plt.xticks([])
                plt.yticks([])

            plt.tight_layout()
            try:
                plt.savefig(f'{Model_name}_checkpoint/random/Key Search-{key}.png')
            except: 
                pattern = r'[\\/\'"@#$%^&*()_+={}[\]:;<>,.?~`!]'    
                # 使用 re.sub 函數替換匹配的字元
                nm = re.sub(pattern, '', key)
                # os.makedirs(f'{Model_name}_checkpoint/random/Key Search-{key}', exist_ok=True)
                plt.savefig(f'{Model_name}_checkpoint/random/Key Search-{nm}.png')
            
        break