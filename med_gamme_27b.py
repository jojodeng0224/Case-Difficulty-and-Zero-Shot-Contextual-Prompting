# pip install accelerate
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import requests
import torch
import argparse
import os
from typing import Dict, List, Any
import json
import textwrap # 在文件开头导入

patho_sign = {"Atelectasis":
                    """ 
                    Direct signs of atelectasis 
                    - displacement of interlobar fissures 
                    - crowding together of pulmonary vessels 
                    - crowded air bronchograms (does not apply to all types of atelectasis; can be seen in subsegmental atelectasis due to small peripheral bronchi obstruction, usually by secretions; if the cause of the atelectasis is central bronchial obstruction, there will usually be no air bronchograms) 
                    Indirect signs of atelectasis 
                    - pulmonary opacification 
                    - shifting granuloma (or any other previously documented lesion, used as a reference for comparison) 
                    - compensatory hyperexpansion of the surrounding or contralateral lung 
                    - displacement of the heart, mediastinum, trachea, hilum 
                    - elevation of the diaphragm 
                    - propinquity of the ribs 

                    Resorptive (obstructive) atelectasis 
                    - increased density (opacity) of the atelectatic portion of lung 
                    - displacement of the fissures toward the area of atelectasis 
                    - upward displacement of hemidiaphragm ipsilateral to the side of atelectasis 
                    - crowding of pulmonary vessels and bronchi in region of atelectasis 
                    - +/- compensatory overinflation of unaffected lung 
                    - +/- displacement of thoracic structures (if atelectasis is substantial) 

                    Linear (plate, discoid, subsegmental) atelectasis 
                    - relatively thin, linear densities in the lung bases oriented parallel to the diaphragm (known as Fleischner lines) 
                    """,
                    "Cardiomegaly": "In most cases, merely 'eye-balling' a chest x-ray will be sufficient in detecting cardiomegaly (as the heart is either clearly normal in size or clearly abnormally enlarged). In equivocal cases, the cardiothoracic ratio (CTR) can be easily calculated on an erect PA chest x-ray. The CTR measures the width of the cardiac silhouette and the thoracic cavity; Traditionally, a ratio greater than 0.5 has been considered to be abnormal.",
                    "Consolidation": 
                    """ 
                    General features of consolidation on CXR include: 
                    - airspace opacification causing obscuration of pulmonary vessels 
                    - air bronchograms 
                    Specific lobar consolidation can be determined by the location of airspace opacification, pattern and the effect on adjacent structures (silhouette sign): 
                    1.right upper lobe consolidation 
                    • may obscure right paratracheal stripe 
                    • may outline the horizontal fissure below 
                    2.right middle lobe consolidation 
                    • may obscure the right heart border 
                    • may outline the horizontal fissure above 
                    3.right lower lobe consolidation 
                    • may obscure the right hemidiaphragm 
                    4.left upper lobe consolidation 
                    • may obscure the left heart border 
                    • may obscure the left paratracheal stripe 
                    5.left lower lobe consolidation 
                    • may obscure the left hemidiaphragm 
                    • may obscure the descending aorta 
                    It must be remembered that the homogeneity of the consolidation will be influenced by any underlying lung disease. For example, consolidation in background emphysematous lung is very often non-confluent. 
                    Occasionally with complete lobar consolidation, there may be an increased volume of the affected lobe, rather than the more frequent neutral or reduced volume. When the fissures are outwardly convex, the appearance is referred to as the bulging fissure sign. 
                    """,
                    "Pleural effusion":
                    """
                    Chest radiographs are the most commonly used examination to assess for the presence of pleural effusion; however, it should be noted that on a routine erect chest x-ray as much as 250-600 mL of fluid is required before it becomes evident 6. A lateral decubitus projection is most sensitive, able to identify even a small amount of fluid. At the other extreme, supine projections can mask large quantities of fluid. 
                    Chest radiograph (lateral decubitus) 
                    - A lateral decubitus film (obtained with the patient lying on their side, effusion side down, with a cross table shoot through technique) can visualize small amounts of fluid layering against the dependent parietal pleura. 
                    Chest radiograph (erect) 
                    - Both PA and AP erect films are insensitive to small amounts of fluid. Features include: 
                    • blunting of the costophrenic angle 
                    • blunting of the cardiophrenic angle 
                    • fluid within the horizontal or oblique fissures 
                    • eventually, a meniscus will be seen, on frontal films seen laterally and gently sloping medially (note: if a hydropneumothorax is present, no such meniscus will be visible) 
                    • with large volume effusions, mediastinal shift occurs away from the effusion (note: if coexistent collapse dominates then mediastinal shift may occur towards the effusion) 
                    - Lateral films are able to identify a smaller amount of fluid as the costophrenic angles are deepest posteriorly. 
                    A subpulmonic effusion (a.k.a. infrapulmonary effusion) may be seen when there is previously established pulmonary disease, but can also be encountered in normal lungs. It can be difficult to identify on frontal radiographs. They are more common on the right, and usually unilateral. The following features are helpful 6: 
                    • right: peak of the hemidiaphragm is shifted laterally 
                    •left: increased distance between lower lobe air and gastric bubble 
                    A lateral decubitus film is again ideal. 
                    Chest radiograph (supine) 
                    - Large amounts of fluid can be present on supine films with minimal imaging changes, as the fluid is dependent and collects posteriorly. There is no meniscus, and only a veil-like increased density of the hemithorax may be visible. It is therefore especially difficult to identify similar sized bilateral effusions as the density of the lungs will be similar. 
                    """,
                    "Pneumothorax":
                    """
                    A pneumothorax is, when looked for, usually easily appreciated on erect chest radiographs. Typically they demonstrate: 
                    - visible visceral pleural edge is seen as a very thin, sharp white line 
                    - no lung markings are seen peripheral to this line 
                    - peripheral space is radiolucent compared to the adjacent lung 
                    - lung may completely collapse 
                    - mediastinum should not shift away from the pneumothorax unless a tension pneumothorax is present (discussed separately) 
                    - subcutaneous emphysema and pneumomediastinum may also be present 
                    """,
                    "Lung mass":
                    """
                    On an x-ray, lung mass may appear as hyperdense pulmonary mass or cavitating pulmonary mass. The characteristic findings of lung mass include: 
                    Rounded or spiculated mass 
                    Bulky hilum (representing the tumor and local nodal involvement) 
                    Lobar collapse 
                    Cavitation may be seen as an air-fluid level 
                    """
                   }

def build_messages(disease_name: str, task: str, patho_sign: Dict[str, str], image: Image.Image) -> List[Dict[str, Any]]:
    """
    将你的 5 类 prompt 重构为 MedGemma 推荐的多模态 messages 结构。

    返回示例：
    [
      {"role": "system", "content": [{"type":"text","text":"..."}]},
      {"role": "user", "content": [{"type":"text","text":"..."}, {"type":"image","image": image}]}
    ]
    """
    # 基础的 system 角色
    system_prompt = "You are an expert radiologist."

    messages: List[Dict[str, Any]] = [
        # {
        #     "role": "system",
        #     "content": [{"type": "text", "text": system_prompt}],
        # }
    ]

    # 构造 user 文本
    if task == "cot":
        user_text = (
            "You are an expert diagnostic radiologist in analysing chest X-ray. Your task is to answer whether a condition is present or not in a chest x-ray."
            f"Do you see {disease_name} in the given chest X-ray? "
            "Please think step by step, and explain your thought along the process, then finally reach a conclusion of whether yes or no the condition is present." 
        )

    elif task == "Knowledge_Prompt":
        # 从 patho_sign 中取描述
        desc = patho_sign.get(disease_name, "")
        desc = textwrap.dedent(desc).strip()

        user_text = (
                "You are a resident radiologist. Your task is to answer whether a condition is present or not in a chest x-ray then finally reach a conclusion of whether yes or no the condition is present."
                f"The following describes possible chest X-ray findings for {disease_name}: {desc} "
                "Based on this description combined with your own knowledge, "
                f"does the following chest X-ray show {disease_name}? "
                "Answer 'yes' if you can see it, or 'no' otherwise."
            )

    elif task == "simple_prompt":
        user_text = (
            "You are an expert diagnostic radiologist in analysing chest X-ray. Your task is to answer whether a condition is present or not in a chest x-ray."
            f"do you see {disease_name} in the given chest X-ray? "
            'Answer "yes" if you can see it, or "no" otherwise.'
        )

    elif task == "instruction":
        # 将“规则/流程”放到 system，有助于模型稳定遵循
        user_text = (
            """
            Role: You are a radiologist who is an expert in interpreting chest x-ray using the ABCDEFGHI approach as the following:"
            ABCDEFGHI approach: 
            A: Assessment of the quality of the image with the following: 
            1. position: is this a supine AP, PA or lateral view? 
            2. inspiration: is this a full inspiration or expiration view? Is the inspiration adequate? 
            3. penetration: is the penetration adequate? 
            4. rotation: is the rotation adequate examined by the the space between the medial clavicle and the margin of the adjacent vertebrae? 
            5. are there any other abnormalities not mentioned?
            B: Bones and soft tissues: 
            1. Bones: are there any fractures, lytic lesions, asymmetry, osteoporosis, or other abnormalities? 
            2. Soft tissues: are there any foreign bodies, swelling, and subcutaneous air? 
            3. are there any other abnormalities not mentioned? 
            C: Cardiac silhouette: 
            1. Is the cardiac silhouette normal or abnormal? 
            2. Is the heart size normal or abnormal? 
            3. Is the heart shape normal or abnormal? 
            4. Is there any calcifications or prosthetic valves or other abnormalities? 
            5. are there any other abnormalities not mentioned? 
            D: Diaphragm: 
            1. Is the position of the diaphrgm normal (the right is commonly slightly higher than the left due to the liver)? 
            2. Is there any elevation, depression, or other abnormalities? 
            3. are there any free gas under the diaphragm? 
            4. are there any other abnormalities not mentioned? 
            E: Effusion: 
            1. Is there any pleural effusion suggested by the costophrenic angle? 
            2. Is there any pericardial effusion? 
            3. are there any other abnormalities not mentioned? 
            F: Fields: 
            1. Are there any infiltrates, consolidation, nodules, pneumothoraces, abnormal vascular markingsss masses, or other abnormalities? 
            2. Are there any 
            G: Great vessels/Gastric bubble: 
            1. Is the aortic size and shape normal? 
            2. Is the aortic knob clearly seen? 
            3. Are there any aortic aneurysm, dissection, or other abnormalities? 
            4. Is the gastric bubble clear and not displaced? 
            5. are there any other abnormalities not mentioned? 
            H:Hila and mediastinum: 
            1. Are there any hilar lymphadenopathy, masses, calcifications or other abnormalities? 
            2. Is the left hilum higher than the right hilum? 
            3. Is there widening of the mediastinum? 
            4. Is there tracheal deviation? 
            5. are there any other abnormalities not mentioned? 
            Task: 
            you will be asked whether you see a condition in the chest x-ray. You should interpret the whole chest x-ray using the ABCDEFGHI approach. After that you should answer the whether you think the asked condition is present with either yes or no.
            """
            f"do you see {disease_name} in the given chest-xray? Answer 'yes' if you can see, or 'no' otherwise."
        )
        # messages[0]["content"].append({"type": "text", "text": instruction})
        # user_text = (
        #     f"Please review the chest X-ray and determine whether {disease_name} is present. "
        #     "After your ABCDEFGHI analysis, answer with 'yes' or 'no'. Answer:"
        # )

    else:  # "unprompt"
        user_text = (
            f"do you see {disease_name} in the given chest X-ray? "
            'Answer "yes" if present, or "no" otherwise.'
        )


    # print("image", image)
    # exit()

    # 组装 user 内容（文本 + 图像）
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "image", "image": image},
            ],
        }
    )

    return messages

patho_sign = {"Atelectasis":
                    """ 
                    Direct signs of atelectasis 
                    - displacement of interlobar fissures 
                    - crowding together of pulmonary vessels 
                    - crowded air bronchograms (does not apply to all types of atelectasis; can be seen in subsegmental atelectasis due to small peripheral bronchi obstruction, usually by secretions; if the cause of the atelectasis is central bronchial obstruction, there will usually be no air bronchograms) 
                    Indirect signs of atelectasis 
                    - pulmonary opacification 
                    - shifting granuloma (or any other previously documented lesion, used as a reference for comparison) 
                    - compensatory hyperexpansion of the surrounding or contralateral lung 
                    - displacement of the heart, mediastinum, trachea, hilum 
                    - elevation of the diaphragm 
                    - propinquity of the ribs 

                    Resorptive (obstructive) atelectasis 
                    - increased density (opacity) of the atelectatic portion of lung 
                    - displacement of the fissures toward the area of atelectasis 
                    - upward displacement of hemidiaphragm ipsilateral to the side of atelectasis 
                    - crowding of pulmonary vessels and bronchi in region of atelectasis 
                    - +/- compensatory overinflation of unaffected lung 
                    - +/- displacement of thoracic structures (if atelectasis is substantial) 

                    Linear (plate, discoid, subsegmental) atelectasis 
                    - relatively thin, linear densities in the lung bases oriented parallel to the diaphragm (known as Fleischner lines) 
                    """,
                    "Cardiomegaly": "In most cases, merely 'eye-balling' a chest x-ray will be sufficient in detecting cardiomegaly (as the heart is either clearly normal in size or clearly abnormally enlarged). In equivocal cases, the cardiothoracic ratio (CTR) can be easily calculated on an erect PA chest x-ray. The CTR measures the width of the cardiac silhouette and the thoracic cavity; Traditionally, a ratio greater than 0.5 has been considered to be abnormal.",
                    "Consolidation": 
                    """ 
                    General features of consolidation on CXR include: 
                    - airspace opacification causing obscuration of pulmonary vessels 
                    - air bronchograms 
                    Specific lobar consolidation can be determined by the location of airspace opacification, pattern and the effect on adjacent structures (silhouette sign): 
                    1.right upper lobe consolidation 
                    • may obscure right paratracheal stripe 
                    • may outline the horizontal fissure below 
                    2.right middle lobe consolidation 
                    • may obscure the right heart border 
                    • may outline the horizontal fissure above 
                    3.right lower lobe consolidation 
                    • may obscure the right hemidiaphragm 
                    4.left upper lobe consolidation 
                    • may obscure the left heart border 
                    • may obscure the left paratracheal stripe 
                    5.left lower lobe consolidation 
                    • may obscure the left hemidiaphragm 
                    • may obscure the descending aorta 
                    It must be remembered that the homogeneity of the consolidation will be influenced by any underlying lung disease. For example, consolidation in background emphysematous lung is very often non-confluent. 
                    Occasionally with complete lobar consolidation, there may be an increased volume of the affected lobe, rather than the more frequent neutral or reduced volume. When the fissures are outwardly convex, the appearance is referred to as the bulging fissure sign. 
                    """,
                    "Effusion":
                    """
                    Chest radiographs are the most commonly used examination to assess for the presence of pleural effusion; however, it should be noted that on a routine erect chest x-ray as much as 250-600 mL of fluid is required before it becomes evident 6. A lateral decubitus projection is most sensitive, able to identify even a small amount of fluid. At the other extreme, supine projections can mask large quantities of fluid. 
                    Chest radiograph (lateral decubitus) 
                    - A lateral decubitus film (obtained with the patient lying on their side, effusion side down, with a cross table shoot through technique) can visualize small amounts of fluid layering against the dependent parietal pleura. 
                    Chest radiograph (erect) 
                    - Both PA and AP erect films are insensitive to small amounts of fluid. Features include: 
                    • blunting of the costophrenic angle 
                    • blunting of the cardiophrenic angle 
                    • fluid within the horizontal or oblique fissures 
                    • eventually, a meniscus will be seen, on frontal films seen laterally and gently sloping medially (note: if a hydropneumothorax is present, no such meniscus will be visible) 
                    • with large volume effusions, mediastinal shift occurs away from the effusion (note: if coexistent collapse dominates then mediastinal shift may occur towards the effusion) 
                    - Lateral films are able to identify a smaller amount of fluid as the costophrenic angles are deepest posteriorly. 
                    A subpulmonic effusion (a.k.a. infrapulmonary effusion) may be seen when there is previously established pulmonary disease, but can also be encountered in normal lungs. It can be difficult to identify on frontal radiographs. They are more common on the right, and usually unilateral. The following features are helpful 6: 
                    • right: peak of the hemidiaphragm is shifted laterally 
                    •left: increased distance between lower lobe air and gastric bubble 
                    A lateral decubitus film is again ideal. 
                    Chest radiograph (supine) 
                    - Large amounts of fluid can be present on supine films with minimal imaging changes, as the fluid is dependent and collects posteriorly. There is no meniscus, and only a veil-like increased density of the hemithorax may be visible. It is therefore especially difficult to identify similar sized bilateral effusions as the density of the lungs will be similar. 
                    """,
                    "Pneumothorax":
                    """
                    A pneumothorax is, when looked for, usually easily appreciated on erect chest radiographs. Typically they demonstrate: 
                    - visible visceral pleural edge is seen as a very thin, sharp white line 
                    - no lung markings are seen peripheral to this line 
                    - peripheral space is radiolucent compared to the adjacent lung 
                    - lung may completely collapse 
                    - mediastinum should not shift away from the pneumothorax unless a tension pneumothorax is present (discussed separately) 
                    - subcutaneous emphysema and pneumomediastinum may also be present 
                    """,
                    "Mass":
                    """
                    On an x-ray, lung mass may appear as hyperdense pulmonary mass or cavitating pulmonary mass. The characteristic findings of lung mass include: 
                    Rounded or spiculated mass 
                    Bulky hilum (representing the tumor and local nodal involvement) 
                    Lobar collapse 
                    Cavitation may be seen as an air-fluid level 
                    """
                   }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run detection/classification and save JSON responses.")
    parser.add_argument("--cls_patho", type=str, required=True, help="Root directory of images.")
    parser.add_argument("--task", type=str, required=True, help="Root directory of images.")
    parser.add_argument("--dataset", type=str, help="Dataset name.")
    args = parser.parse_args() 
    # out_png = "loaded_check.png"

    # model_id = "google/medgemma-4b-it"
    model_id = "google/medgemma-27b-it"
    cache_dir = ""
    
    model_save_name = model_id.split("/")[-1]

    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=cache_dir
    )
    processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)


    output_json = f"//{args.dataset}_{args.cls_patho}_{args.task}_{model_save_name}.json"
    classes = ["atelectasis",  "cardiomegaly",  "consolidation",  "effusion",  "mass",  "normal",  "pneumothorax"]
    results = [] 
    
    # print(f'Image paths: {image_paths}')
    for cls in classes:
        if args.dataset == "padchest":
            image_paths = ""
        else:
            image_paths = ""
        
        image_paths = os.path.join(image_paths, cls)
        # image = Image.open(image_paths)
        for dir in os.listdir(image_paths):
            # print(f'Processing disease folder: {dir}')
            if dir.startswith('.'):
                continue  # Skip hidden files/directories

            disease_folder = os.path.join(image_paths, dir)
            # print("dir", dir)
            
            # target_images = {
            #     # "00000148_001.png": "Atelectasis",
            #     # "00000090_003.png": "Atelectasis",
            #     "00000865_000.png": "Cardiomegaly"
            # }
            # if dir not in target_images:
            #     continue # 跳过所有其他图片
            
            # query_disease = target_images[dir]
            # =======================================================

            # image = Image.open(disease_folder)
            # print(f'Processing disease folder path: {disease_folder}')
            
            # exit()
            print(f'Processing disease folder path: {disease_folder}')
            image = Image.open(disease_folder)
            # print("image1111", image)

            # arr = np.array(image)  
            # plt.figure(figsize=(6, 6))
            # if arr.ndim == 2:
            #     # For 16-bit images, normalize for visualization
            #     if arr.dtype == np.uint16 or arr.dtype == np.int16:
            #         arr_vis = (arr.astype(np.float32) - arr.min()) / max(1, (arr.max() - arr.min()))
            #         plt.imshow(arr_vis, cmap='gray')
            #     else:
            #         plt.imshow(arr, cmap='gray')
            # else:
            #     # If it's 16-bit RGB(A), scale to 0-1 for imshow
            #     if arr.dtype == np.uint16:
            #         plt.imshow(arr / 65535.0)
            #     else:
            #         plt.imshow(arr)
            # plt.axis('off')
            # plt.tight_layout()
            # plt.savefig(out_png, dpi=150)
            # plt.close()
            # exit()
            messages = build_messages(args.cls_patho, args.task, patho_sign, image)


    # # Image attribution: Stillwaterising, CC0, via Wikimedia Commons
    # image_url = "https://upload.wikimedia.org/wikipedia/commons/c/c8/Chest_Xray_PA_3-8-2010.png"
    # image = Image.open(requests.get(image_url, headers={"User-Agent": "example"}, stream=True).raw)

    # messages = [
    #     {
    #         "role": "system",
    #         "content": [{"type": "text", "text": "You are an expert radiologist."}]
    #     },
    #     {
    #         "role": "user",
    #         "content": [
    #             {"type": "text", "text": "Describe this X-ray"},
    #             {"type": "image", "image": image}
    #         ]
    #     }
    # ]

            inputs = processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt"
            ).to(model.device, dtype=torch.bfloat16)

            input_len = inputs["input_ids"].shape[-1]

            with torch.inference_mode():
                generation = model.generate(**inputs, max_new_tokens=2048, temperature=0.0, do_sample=False)
                generation = generation[0][input_len:]

            decoded = processor.decode(generation, skip_special_tokens=True)
            # print(decoded)
            results.append({
                    "class": cls,
                    "dir": dir,
                    "path": disease_folder,
                    "prompt": messages[0]["content"][0]["text"],
                    "response": decoded,
                    })
            # print("" + "=" * 80 + "\n")
            # print("results", results)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(results)} responses to {output_json}")
