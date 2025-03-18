import pickle
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from tqdm import tqdm
import numpy as np
import torch
import json
import argparse
from models.embeddingModels import *
import logging
import asyncio
from tqdm.asyncio import tqdm_asyncio
from openai import AsyncOpenAI
from concurrent.futures import ThreadPoolExecutor
os.chdir('./CPGRec_plus')

url_embed = "https://api.xxx.cn/v1/embedding"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="error_log.log", 
)
logger = logging.getLogger(__name__)



parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-7B-Instruct')
parser.add_argument('--seed', type=int, default=3407)

args = parser.parse_args()


model = args.model
seed = np.random.randint(0, 100000000)

import random
random.seed(seed)



keys = ['sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx',
        'sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx',
        'sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx']
keys = tuple(keys)
model_embed = 'BAAI/bge-m3'


embeddingers = EmbeddingModel(url_embed, keys, model_embed)


def compute_embeddings(texts_batch):
    return embeddingers.get_embedding(texts_batch, batch_size=32)




list_clients = [
            AsyncOpenAI(
                api_key=api_key,
                base_url="https://api.xxx.cn/v1/chat/completions",
            )
            for api_key in keys  
]


list_models = [
    'Qwen/Qwen2.5-7B-Instruct',
]











def update_item_info(dic, t, r):
    dic = json.loads(dic)
    t, r = np.round(t.item(), 1), np.round(r.item(), 1)
    dic.update({'personal interest': t, 'global interest': r})

    return dic


class Counter:
    def __init__(self):
        self._count = 0

    def next(self):
        self._count += 1
        return self._count-1

    def current(self):
        return self._count



def get_user_embedding_with_LLM(model = model, model_embed = 'BAAI/bge-m3'):

    path_train_data = './data_exist/train_data.pkl'
    with open(path_train_data, 'rb') as f:
        dic_hist_info = pickle.load(f)

    dic_hist_info = {u: tuple(items) for u, items in list(dic_hist_info.items())}
    path_user_prompts = './data_exist/user_prompts.pkl'

    if os.path.exists(path_user_prompts):
        try:
            with open(path_user_prompts, 'rb') as f:
                user_prompts = pickle.load(f)

        except:
            print("error occurs when loading user promps.")
            exit(0)


    else:
        path_item_profs = './CPGRec_plus/data_exist/item_profs.pkl'
        with open(path_item_profs, 'rb') as f:
            item_profs = pickle.load(f)

        path_t, path_r = "./CPGRec_plus/data_exist/t.pth", "./CPGRec_plus/data_exist/r.pth"
        t, r = torch.load(path_t), torch.load(path_r)
        t, r = (t - t.mean())/t.std(), (r - r.mean())/r.std()
    

        user_prompts = {}
        g_i = Counter() 



        for u in tqdm(dic_hist_info.keys()):
            dic_for_u = {
                'information of historical games': {
                    f'historical game {i}': update_item_info(item_profs[i], t[g_i.current()], r[g_i.next()])
                    for i in dic_hist_info[u] 
                }
            }

            dic_data = dic_for_u['information of historical games']
            abs_t_r = np.abs([dic_data[_]['personal interest'] - dic_data[_]['global interest'] for _ in dic_data.keys()])
            indices_neg = np.argsort(abs_t_r)[2:-2]
            mask_pos = np.ones(len(abs_t_r)).astype(bool)
            mask_pos[indices_neg] = False
            
            dic_data = {key: value for _, (key, value) in enumerate(dic_data.items()) if (value is not None) and (mask_pos[_])}
            dic_for_u['information of historical games'] = dic_data
            user_prompts[u] = str(dic_for_u)
        with open(path_user_prompts, 'wb') as f:
            pickle.dump(user_prompts, f)


    instruct_prompt = r"You will serve as an assistant to profile a player's personal interest based on meticulous comparison between his or her personal interest and global interest, and your rich knowledge about the video games. I will provide you with the historical gaming records of each player along with his or her personal interest for each according game and the global interest, which represents the interest of general players.\
                    Here are some instructions:\
                    1. The provided information of this player is in the form of a JSON string:\
                        {'information of historical games': 'the historical gaming records of a player'},\
                    2. To be more detailed, the aforementioned historical records of a player is in the form of a JSON string:\
                        {'historical game i': {'description':'description for game i', 'reasoning':'explaination for the generation of description for game i',\
                            'personal interest':'a float number represents the personal interest of this player for game i, which follows the standard normal distribution', (A higher value indicates a greater degree of personal interest).'\
                                'global interest':'a float number represents the global interest of general players for game i, which follows the standard normal distribution', (A higher value indicates a greater degree of personal interest).'}\
                        {'historical game j': {... (which is the same as the aforementioned case of game i)}}, ...(Similarily, all the video games played by this player will be included)}\
                    Requirements:\
                    1. Please provide your answer in a JSON format, following this structure:\
                        {'description':'the most valuable description for this player based on meticulous comparison between his or her personal interest and global interest along with your knowledge about the historical video games',\
                        'reasoning':'briefly explain your reasoning for the description, taking the meticulous person-globality comparison as the most important factor'}\
                    2. Please ensure that the 'description' is no longer than 200 words.\
                    3. Please ensure that the 'reasoning' is no longer than 100 words.\
                    4. Please do not provide any text within your answer in any form out of JSON string.\
                    5. Your answer must be in English.\
                    6. As for the comparison between this player's personal interest and the general players' global interest: A personal interest sigfinicantly\
                        larger than global interest well reflects this particular player's personal interest, while one smaller than global interest obviously shows his or her disinterest.\
                        The knowledge of historical games that this player has shown a significant personal interest in serves as more crucial information for the profiling of this player,\
                        in contrast, the information of games played that this player has shown witnessed disinterest also effectively reflects the personal preference of this player from an opposite side."







    path_user_profs = './CPGRec_plus/data_exist/user_profs.json'
    if os.path.exists(path_user_profs):
        with open(path_user_profs, 'r') as f:
            user_profs = json.load(f)

        if len(list(user_profs.keys()))<len(dic_hist_info.keys()): go = False
        else: go = False

    if go:
        async def process_user(instruct_prompt, user_prompt, user_id):
            client = random.choice(list_clients)

            try:
                response = await client.chat.completions.create(
                    model=random.choice(list_models),
                    messages=[
                        {"role": "system", "content": instruct_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0,
                    max_tokens=128
                )
                return user_id, response.choices[0].message.content
            except Exception as e:
                logger.error(f"Error processing user {user_id}: {e}")
                logger.error(f"User prompt: {user_prompt}")
                logger.error(f"System prompt: {instruct_prompt}")
                raise  

        async def batch_process_users(dic_hist_info, instruct_prompt, max_concurrent, delay, n, path_user_profs, user_prompts):
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def limited_task(user_id, prompt):
                async with semaphore:
                    result = await process_user(instruct_prompt, prompt, user_id)
                    await asyncio.sleep(delay) 
                    return result
            
            if os.path.exists(path_user_profs):
                with open(path_user_profs, 'rb') as f:
                    results = pickle.load(f)
            else:
                results = {}
            

            
            
            #all user included
            path_remaining_users = './CPGRec_plus/data_exist/remaining_users.pkl'
            remaining_users = [user_id for user_id in user_prompts.keys() if user_id not in results]

            while len(remaining_users)>0:
                
                print(f'{len(remaining_users)}/{len(dic_hist_info.keys())}')

                try:
                    with open(path_remaining_users, 'rb') as f:
                        remaining_users = pickle.load(f)
                except:
                    pass

                batch_users = random.sample(remaining_users, n)
                batch_tasks = [limited_task(user_id, user_prompts[user_id]) for user_id in batch_users]


                try:
                    batch_results = await asyncio.wait_for(
                        tqdm_asyncio.gather(*batch_tasks),
                        timeout=65 
                    )
                    results.update({user_id: result for user_id, result in batch_results})
                    
                    with open(path_user_profs, 'wb') as f:
                        pickle.dump(results, f)

                    with open(path_remaining_users, 'wb') as f:
                        remaining_users = [user_id for user_id in user_prompts.keys() if user_id not in results]
                        pickle.dump(remaining_users, f)

                    await asyncio.sleep(30)
                except asyncio.TimeoutError:
                    error_info = f"Batch processing timed out."
                    print(error_info)
                    logger.error(error_info)
                    continue  
                except Exception as e:
                    error_info = f"Error in batch processing: {e}"
                    print(error_info)
                    logger.error(error_info)
                    continue 

            return results

        user_profs = asyncio.run(batch_process_users(dic_hist_info, instruct_prompt, max_concurrent=20*15, delay=0, n=600, path_user_profs=path_user_profs, user_prompts=user_prompts))












# sleep 15; timeout 65; max_concurrent 20*10; n 600

    user_profs = {int(k): v for k, v in user_profs.items()}
    empty_users = list(set(user_prompts.keys()) - set(user_profs.keys()))




    # for i in user_profs.keys():
    #     try:
    #         user_profs[i] = json.loads(user_profs[i])
    #     except Exception as e:
    #         print()
    
    # indices_none = np.where(np.array([user_profs[i]['description'] for i in user_profs.keys()])=='None')[0]



    # for i in user_profs.keys():
    #     user_profs[i] = str(user_profs[i])




    texts = list(np.zeros(len(user_prompts.keys())))
    for k in tqdm(user_prompts.keys()):
        if k in user_profs.keys():
            texts[k] = user_profs[k]
        else:
            texts[k] = user_prompts[k]



    path_user_emb = './CPGRec_plus/data_exist/user_emb.pth'
    if os.path.exists(path_user_emb):
        embeddings = torch.load(path_user_emb)
    else:
        # embeddings = embeddingers.get_embedding(texts, batch_size=32)

        n = 3908744
        ls_p = np.linspace(0, n, 20).astype(int)
        p_pre, p_tail = ls_p[:-1], ls_p[1:]
        batches = [texts[p_pre[i]:p_tail[i]] for i in range(len(p_pre))]


        # ls_embeddings = [embeddingers.get_embedding(texts[p_pre[i]:p_tail[i]], batch_size=32) for i in range(len(p_pre))]
        # torch.save(embeddings, path_user_emb)

        with ThreadPoolExecutor() as executor:
            ls_embeddings = list(executor.map(compute_embeddings, batches))

        embeddings = torch.concat(ls_embeddings)

        torch.save(embeddings, path_user_emb)



    return embeddings



if __name__ == "__main__":
    get_user_embedding_with_LLM()