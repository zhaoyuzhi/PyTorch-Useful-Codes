import json

# save a dictionary object to a json file
def json_save(json_content, json_file):
    with open(json_file, 'a') as fp:
        json.dump(json_content, fp, indent = 4)
    
# read a json file, return the json_content
def json_read(json_file):
    with open(json_file, 'r') as f:
        # json_content = json.load(f)
        json_content = json.loads(f.read())
        return json_content

'''
json load/loads是将json格式的数据转成python对象，简单来说
load是对文件进行操作的，因此load的主要参数是打开的文件，调用格式通常为 load(f)
loads是对字符串进行操作的，因此loads的主要参数是字符串，调用格式通常为 load(str)
（为了方便记忆，可以把loads后面的小尾巴s理解为str）

dump是将python对象转成json格式存入文件，主要格式是dump(obj, f)
dumps是将python对象转成json格式的字符串，主要格式是dumps(obj)
'''

if __name__ == '__main__':

    # -----------------------------------------------------------
    '''
    json_list = json_read("filtered_val_list.json")
    print(len(json_list))
    print(json_list[0])
    '''
    
    # -----------------------------------------------------------
    img_json = {}
    img_json["image"] = "name"
    img_json["conversations"] = [
        {
            "from": "human",
            "value": "llm_output_line_question" + "\n<image>"
        },
        {
            "from": "gpt",
            "value": "llm_output_line_answer"
        }
    ]
    print(img_json)
    json_save(img_json, 'test.json')
