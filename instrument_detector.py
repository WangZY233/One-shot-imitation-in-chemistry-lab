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

if __name__ == '__main__':
    
    # Path to your image
    image_path = "./notebooks/images/test2.jpg"

    # Get the instrument category
    instrument_category = get_instrument_category(image_path)
    