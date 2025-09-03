from openai import OpenAI
import base64
from PIL import Image
import tqdm,json,os

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# 创建客户端时指定自定义的 base URL
client = OpenAI(
        base_url="https://api.gptsapi.net/v1",
        api_key="your key"
    )

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

# Path to your image
video_list = os.listdir("/root/autodl-tmp/vip200k_test_track_Facial")
# use_dict
data = {}

for video in tqdm.tqdm(video_list):
    if '.' in video:
        continue
    video_image_dir = "/root/autodl-tmp/vip200k_test_track_Facial/"+video+"/"
    image_path = video_image_dir+"image.png"
    data[video] = {}
    
    with Image.open(image_path) as img:
        width, height = img.size
        
    # Getting the base64 string
    base64_image = encode_image(image_path)
    
    for prompt_id in range(1,6):
        with open( "/root/autodl-tmp/vip200k_test_track_Facial/"+video+"/prompt"+str(prompt_id)+'.txt', 'r') as file:
                video_description = file.read()

        

# ── 2. 提示词模板（保留 {caption} 占位符） ────────────────
        PROMPT_TEMPLATE = """You will receive:
        1. Original caption: <<<{caption}>>>
        2. An image containing a person’s face.
        
        Your task:
        • Look at the face in the image and distill a brief description that mentions only facial features (e.g., expression, approximate age, gender presentation, notable facial characteristics).
          -- Do NOT include clothing, accessories, background elements, or anything not strictly part of the face.
        • Keep the original caption completely intact—every word, punctuation mark, and capital letter must remain unchanged.
        • Seamlessly insert your facial description into the caption so the result reads naturally as one sentence or paragraph.
        • If the caption already mentions a person, let your description enrich it instead of repeating details.
        • Output **only** the final, new caption—no explanations, labels, quotation marks, or extra text of any kind.
        """
        
        # ── 3. 用 str.format() 插入 caption ─────────────────────
        prompt_filled = PROMPT_TEMPLATE.format(caption=video_description)
        
        # ── 4. 调用 Chat Completions API ───────────────────────
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt_filled          # 已经带入 caption
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=5000
        )
    

        data[video][prompt_id] =   response.choices[0].message.content
        
        with open('prompt_withfacedescription_all.json', 'w', encoding='utf-8') as file:
            file.write(json.dumps(data, indent=2))

