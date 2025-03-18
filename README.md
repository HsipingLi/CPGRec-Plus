## CPGRec+: A Balance-oriented Framework for Personalized Video Game Recommendations

##### Authors: Xiping Li, Aier Yang, Jianghong Ma*, Kangzhe Liu, Shanshan Feng*, Haijun Zhang, Yi Zhao.

##### Abstract: The rapid expansion of gaming industry requires advanced recommender systems tailored to its dynamic landscape. Existing Graph Neural Network (GNN)-based methods primarily prioritizes accuracy over diversity, and overlooks the inherent trade-off between them. To address this, we previously proposed CPGRec, a balance-oriented gaming recommender system. However, CPGRec fails to account for critical disparities in player-game interactions, which carry varying significance in reflecting players' personal preferences and may exacerbate the over-smoothing issues inherent in GNN-based models. Moreover, existing approaches underutilize the reasoning capabilities and extensive knowledge of large language models (LLMs) in addressing these limitations. 
To bridge this gap, we propose two new modules. First, the Preference-informed Edge Reweighting (PER) module assigns signed edge weights to qualitatively distinguish significant player interests and disinterests while then quantitatively measuring preference strength, thereby mitigating over-smoothing in graph convolutions. Second, the Preference-informed Representation Generation (PRG) module leverages LLMs to generate contextualized descriptions of games and players, thereby enriching their representations by incorporating both global and personal interests. Experiments on Steam dataset demonstrate CPGRec+'s superior accuracy and diversity over state-of-the-art models.


## Environment
See file ***requirements.txt*** for details.\
\
***Profile:***\
cuda - 11.3\
dgl - 0.9.1.post1\
numpy - 1.24.4\
torch - 1.12.1\
torchaudio - 0.12.1\
torchvision - 0.13.1\
python - 3.9




## Dataset
Download: https://drive.google.com/file/d/1F9kr_YWimBtexJEH-zkDzCOwl1q7GmFp/view

Please put the downloaded dataset in the folder ***steam_data***.

## To Run the Code
***run.sh***

