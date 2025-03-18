
import pickle
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(__file__))
from tqdm import tqdm
import numpy as np
import torch
os.chdir('./CPGRec_plus')


from models.embeddingModels import EmbeddingModel


def get_item_embedding_with_LLM(model = 'Qwen/Qwen2.5-7B-Instruct', model_embed = 'BAAI/bge-m3'):


    path_app_info = './CPGRec_plus/data_exist/dic_app_info_raw.pkl'
    with open(path_app_info, 'rb') as f:
        dic_info = pickle.load(f)
        print('')


    path_item_prompts = './CPGRec_plus/data_exist/item_prompts.pkl'

    if os.path.exists(path_item_prompts):
        with open(path_item_prompts, 'rb') as f:
            item_prompts = pickle.load(f)
    else:
        item_prompts = {}
        for i in dic_info.keys():
            item_prompts[i] = {
                'title': dic_info[i][0],
                'average rating': f'{dic_info[i][3]:.0f}',
                'supplementary information': f'its price is {dic_info[i][1]} dollars and released time is {dic_info[i][2]}'
            }

            item_prompts[i] = str(item_prompts[i])
        with open(path_item_prompts, 'wb') as f:
            pickle.dump(item_prompts, f)


    import json  
    from openai import OpenAI

    client = OpenAI(
        api_key="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
        base_url="https://api.xxx.cn/v1",
    )


    instruct_prompt = r"You will serve as an assistant with rich knowledge about the video games released on Steam platform, who are required to provide a description pocessing the most key information for each video game. I will provide you with the title of video game, as well as the average rating by players on it.\
                    Here are some instructions:\
                    1. The provided information of video game is in the form of a JSON string that contains its title and average rating:\
                        {'title': 'the title of the video game',\
                        'average rating': 'the average rating (100-scale) of historical players on the video game',\
                        'supplementary information': 'supplementary information of the video game, including its price(US dollar) and released time(Year-Month-Day), (price=0 indicates this game is free)\
                        }\
                    Requirements:\
                    1. Please provide your answer in a JSON format, following this structure:\
                        {'description':'the most valuable information as a description for the video game, taking both its inherent information and the plays' average rating on it as important factors into consideration',\
                        'reasoning':'briefly explain your reasoning for the description taking average rating into account'}\
                    2. Please ensure that the 'description' is no longer than 500 words.\
                    3. You are not required to include the information about the developer, publisher, price and released time into the description.\
                    4. Please ensure that the 'reasoning' is no longer than 200 words.\
                    5. Please do not provide any text within your answer in any form out of JSON string.\
                    6. Your answer must be in English.\
                    7. If the title of video game is null, just leave each term of your answer 'None'"







    path_item_profs = './CPGRec_plus/data_exist/item_profs.pkl'
    if os.path.exists(path_item_profs):
        with open(path_item_profs, 'rb') as f:
            item_profs = pickle.load(f)

    else:
        import asyncio
        from tqdm.asyncio import tqdm_asyncio
        from openai import AsyncOpenAI

        async def process_item(client, instruct_prompt, item_prompt, item_id):
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": instruct_prompt},
                    {"role": "user", "content": item_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=512
            )
            return item_id, response.choices[0].message.content

        async def batch_process_items(dic_info, instruct_prompt, max_concurrent=5):
            client = AsyncOpenAI(
                    api_key="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                    base_url="https://api.xxx.cn/v1",
            )  
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def limited_task(item_id, prompt):
                async with semaphore:
                    return await process_item(client, instruct_prompt, prompt, item_id)
            
            tasks = [limited_task(i, item_prompts[i]) for i in dic_info.keys()]
            results = await tqdm_asyncio.gather(*tasks)
            
            return {item_id: result for item_id, result in results}

        item_profs = asyncio.run(batch_process_items(dic_info, instruct_prompt))

        with open(path_item_profs, 'wb') as f:
            pickle.dump(item_profs, f)



    import json
    for i in item_profs.keys():
        item_profs[i] = json.loads(item_profs[i])

    indices_none = np.where(np.array([item_profs[i]['description'] for i in item_profs.keys()])=='None')[0]

    for i in item_profs.keys():
        item_profs[i] = str(item_profs[i])

    url = "https://api.xxx.cn/v1/embeddings"
    key = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    model = model_embed

    embeddingers = EmbeddingModel(url, keys = [key], model = model)

    texts = list(item_profs.values())

    path_item_emb = './CPGRec_plus/data_exist/item_emb.pth'
    if os.path.exists(path_item_emb):
        embeddings = torch.load(path_item_emb)
    else:
        embeddings = embeddingers.get_embedding(texts, indices_none)

        mask = torch.zeros(embeddings.shape[0]).bool()
        mask[indices_none] = True
        embeddings[mask] = torch.mean(embeddings[~mask], axis = 0)

        torch.save(embeddings, path_item_emb)

    return embeddings




if __name__ == "__main__":
    get_item_embedding_with_LLM()





