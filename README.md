## CPGRec+: A Balance-oriented Framework for Personalized Video Game Recommendations

##### Authors: Xiping Li, Aier Yang, Jianghong Ma*, Kangzhe Liu, Shanshan Feng*, Haijun Zhang, Yi Zhao.

##### Abstract: The rapid growth of the video game industry has led to an overwhelming number of game choices, creating a pressing need for specialized video game recommender systems. While Graph Convolutional Network (GCN)-based models have gained popularity for user modeling, existing systems often prioritize accuracy over diversity, resulting in repetitive suggestions. Additionally, these models overlook the disparities in observed interactions, which are exacerbated by the smoothness property of GCNs, ultimately compromising accuracy. This study delves into the smoothness property of GCN-based models, identifying opportunities to better model interaction disparities by leveraging dwell time and ratings on video game platforms. Building on these findings, we propose a novel model, CPGRec+, which employs preference-informed edge reweighting to address disparities in historical interactions. The model integrates category and popularity semantics through graphical transformations and incorporates rating-based insights via reweighted BPR loss to achieve balance-oriented representation learning. Furthermore, by analyzing the disparities in historical interactions, CPGRec+ uses the normal-Fisher distribution link to quantify personal preferences. Experiments conducted on the real-world Steam dataset highlight the superior performance of CPGRec+ in delivering more accurate and diverse recommendations.


## Environment
See file ***requirements.txt*** for details.

## Dataset
https://drive.google.com/file/d/1F9kr_YWimBtexJEH-zkDzCOwl1q7GmFp/view

## To Run the Code
***run.sh***

