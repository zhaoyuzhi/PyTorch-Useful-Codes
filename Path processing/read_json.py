import json

# read a json file, return the contents
def get_json_contents(json_file):
    with open(json_file, 'r') as f:
        contents = json.loads(f.read())
        return contents

if __name__ == '__main__':

    json_list = get_json_contents("a_json_file.json")
    print(len(json_list))
    print(json_list[0])
    