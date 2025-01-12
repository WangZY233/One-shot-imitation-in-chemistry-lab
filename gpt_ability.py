import os
import sys
import base64
# 获取parent_folder文件夹的路径
parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# 将parent_folder文件夹添加到sys.path列表中
sys.path.append(parent_path)
sys.path.append("../")
import re
import openai
import cv2

openai.api_key = "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" 
proxy = {
'http': 'http://127.0.0.1:7890',
'https': 'http://127.0.0.1:7890'
}

def execute_code(code):
    try:
        exec(code)
    except Exception as e:
        print(f"Error executing code: {e}")

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    
# 使用gpt获取化学仪器类别
def get_instrument_category(image_path):
    # Getting the base64 string
    base64_image = encode_image(image_path)
    
    # GPT
    openai.proxy = proxy
    file = open('./prompt/detector.txt', 'r')
    promp = file.read()
    conversation = [{"role": "user", "content": promp}]
    
    conversation.append({"role": "user", 
                            "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                            },
                        ],
                        })
    
    completion = openai.ChatCompletion.create(
        model="gpt-4o", 
        messages=conversation
    )
    
    reply = completion.choices[0]['message']['content']
    # reply = completion.choices[0]
    print("Robot:",reply)
    conversation.append({"role": "assistant", "content": reply})

    return reply

# 使用gpt获取动作
def get_movement(image_path):
    # Getting the base64 string
    base64_image = encode_image(image_path)
    
    # GPT
    openai.proxy = proxy
    file = open('./prompt/movement_detector.txt', 'r')
    promp = file.read()
    conversation = []
    
    conversation.append({"role": "user", 
                            "content": [
                            {
                                "type": "text",
                                "text": promp,
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                            },
                        ],
                        })
    
    completion = openai.ChatCompletion.create(
        model="gpt-4o", 
        messages=conversation
    )
    
    reply = completion.choices[0]['message']['content']
    # reply = completion.choices[0]
    print("Robot:",reply)
    conversation.append({"role": "assistant", "content": reply})

    return reply

if __name__ == '__main__':
    SLECTED_FOLDER = "./selected_frames/testvideo"
    frame_names = [
            p for p in os.listdir(SLECTED_FOLDER)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
        ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    for i in range(len(frame_names)):
    #     img = cv2.imread(os.path.join(SLECTED_FOLDER, frame_names[0]))
    # # Path to your image
    # image_path = "./custom_video_frames/00142.jpg"

    # Get the instrument category
    # instrument_category = get_instrument_category(image_path)
        get_movement(os.path.join(SLECTED_FOLDER, frame_names[i]))
    