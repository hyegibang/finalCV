import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from model import load_model, preprocess
import os
import torch
from sklearn import cluster
# import gradio as gr
import torch
from model import preprocess, load_model
from transformers import CLIPModel, CLIPProcessor

#global variables 
n_clusters=500
curr_cluster_idx = 250

def load_embd():
    emd_path = "embeddings/img_emb_0.npy"
    embd_full = np.load(emd_path)
    image_dir = os.listdir("images")
    img_index = sorted(list(map(lambda sub:int(''.join(
          [ele for ele in sub if ele.isnumeric()])), image_dir)))

    embd_data = pd.DataFrame(embd_full)
    clusters = cluster.KMeans(n_clusters).fit(embd_data)
    cluster_map = pd.DataFrame()
    # cluster_map['data_index'] = [img_index[i] for i in embd_data.index.values]
    cluster_map['data_index'] = embd_data.index.values
    cluster_map['label'] = clusters.labels_
    cluster_map.to_pickle("./cluster_map.pkl")
    return cluster_map


#def load_aest_model(): 
    
def predict(img):
    inputs = clip_processor(images=img, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        vision_output = vision_model(**inputs)
    pooled_output = vision_output.pooler_output
    embedding = preprocess(pooled_output)
    with torch.no_grad():
        rating = rating_model(embedding)
        artifact = artifacts_model(embedding)
    return rating.detach().cpu().item(), artifact.detach().cpu().item()

def initialize_cluster_pd(clusters, cluster_map): 
    cluster_pd = pd.DataFrame(columns = ["label","img_idx", "center", "rating_score", "aest_score"])
    cluster_pd["label"] = list(range(n_clusters))
    cluster_pd["img_idx"] = [np.array(cluster_map[cluster_map['label'] == x]["data_index"]) for x in range(n_clusters)]
    cluster_pd["center"] = list(clusters.cluster_centers_)
    cluster_pd.to_pickle("./emd_init.pkl")
    return 

def predict_score_main(image_dir, cluster_pd): 
    for idx in range(curr_cluster_idx + 1, n_clusters):
        print("current cluster: " + str(idx))
        curr_clus_img_list = cluster_pd["img_idx"][idx]
        clus_rating_score = 0  # high is good
        clus_artif_score = 0  # low is good
        for curr_img_idx in curr_clus_img_list:
            error_cnt = 0 
            img_path = "images/" + image_dir[curr_img_idx]
            try: 
                clus_pred = predict(plt.imread(img_path))
            except KeyError: 
                clus_pred = (0,0) 
                error_cnt += 1
            except TypeError: 
                clus_pred = (0,0)
                error_cnt += 1
            clus_rating_score += int(clus_pred[0])
            clus_artif_score += int(clus_pred[1])
        cluster_pd.loc[idx, "rating_score"] = clus_rating_score / (len(curr_clus_img_list) - error_cnt)
        cluster_pd.loc[idx, "aest_score"] = clus_artif_score / (len(curr_clus_img_list) - error_cnt) 
        if (idx % 50 == 0 or idx == 499):
            cluster_pd.to_pickle("./full_emd_score" + str(idx) + ".pkl")
        
if __name__ == "__main__":
    cluster_pd = pd.read_pickle("./full_emd_score" + str(curr_cluster_idx) + ".pkl")
    print(cluster_pd.iloc[curr_cluster_idx])
    image_dir = os.listdir("images")
    model = load_model("models/aesthetics_scorer_rating_openclip_vit_h_14.pth")
    MODEL = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    # DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    DEVICE = 'cuda'
    
    model = CLIPModel.from_pretrained(MODEL)
    vision_model = model.vision_model
    vision_model.to(DEVICE)
    del model
    clip_processor = CLIPProcessor.from_pretrained(MODEL)
    
    rating_model = load_model("models/aesthetics_scorer_rating_openclip_vit_h_14.pth").to(DEVICE)
    artifacts_model = load_model("models/aesthetics_scorer_artifacts_openclip_vit_h_14.pth").to(DEVICE)
    
    predict_score_main(image_dir, cluster_pd)
    
