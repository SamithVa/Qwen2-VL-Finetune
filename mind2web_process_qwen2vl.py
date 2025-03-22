import json
import os
from tqdm import tqdm
import random
import argparse
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--imgs_dir', type=str, default='/data/data1/syc/intern/wanshan/mind2map_dataset/mind2web_images')
parser.add_argument('--annot_train_json', type=str, default="/data/data1/syc/intern/wanshan/mind2map_dataset/mind2web_data_train.json")
args = parser.parse_args()


# convert action to prediction format
def action2step(action, image_size):
    action_type = action["operation"]["original_op"]
    assert action_type in ['CLICK', 'TYPE', 'SELECT', 'HOVER', 'ENTER']  # five types of data

    point_x = action["bbox"]["x"] + (action["bbox"]["width"] / 2)
    point_y = action["bbox"]["y"] + (action["bbox"]["height"] / 2)
    click_point = [point_x / image_size[0], point_y / image_size[1]]
    click_point = [round(item, 3) for item in click_point]
    click_point = [f"{item:.2f}" for item in click_point]
    click_point = "({},{})".format(click_point[0], click_point[1])

    if action_type in ['CLICK', 'HOVER', 'ENTER']:  # following mind2web, these three actions are regarded as click
        action_step = "{{\"action_type\": {}, \"click_point\": {}}}".format(4, click_point)
    elif action_type == 'SELECT':
        select_value = action["operation"]["value"]
        action_step = "{{\"action_type\": {}, \"click_point\": {}, \"value\": \"{}\"}}".format(2, click_point, select_value)
    elif action_type == 'TYPE':
        typed_text = action["operation"]["value"]
        action_step = "{{\"action_type\": {}, \"click_point\": {}, \"value\": \"{}\"}}".format(3, click_point, typed_text)
    return action_step


mind2web_imgs_dir = args.imgs_dir
mind2web_train = json.load(open(args.annot_train_json, 'r'))
train_step = []
prompt_origin = "Please generate the next move according to the ui screenshot, instruction and previous actions. Instruction: {}. Previous actions: {}"
step_i = 0
for episode in tqdm(mind2web_train):
    goal = episode["confirmed_task"]
    annot_id = episode["annotation_id"]
    previous_actions = []

    for idx, step in enumerate(episode["actions"]):

        # Few actions can not find its corresponding bbox, jump these actions
        if "bbox" not in step:
            print("action not found")
            continue

        image_filename = annot_id + '-' + step["action_uid"] + '.jpg'
        img_path = os.path.join(mind2web_imgs_dir, image_filename)
        if not os.path.exists(img_path):
            print("img not found")
            input()
        image = Image.open(img_path)

        # visualize step data
        # show_image_with_bbox(image, step["bbox"])
        # print(step)
        # input()

        previous_step = ""
        for i, action in enumerate(previous_actions[-4:idx]):
            previous_step += 'Step' + str(i) + ': ' + action + ". "

        action_step = action2step(step, image.size)
        previous_actions.append(action_step)

        prompt = prompt_origin.format(goal, previous_step)

        conversations = []
        image = image_filename
        conv_human = {
            "from": "human", 
            "value": f"<image>\n{prompt}"}
        conv_gpt = {
            "from": "gpt", 
            "value": action_step}
        conversations.extend([conv_human, conv_gpt])

        train_step.append({
                        "id": "mind2web_{}".format(step_i), 
                        "conversations": conversations, 
                        "image": image
                        })
        step_i += 1

random.shuffle(train_step)
print("Num of total step: " + str(len(train_step)))
json.dump(train_step, open("./data/mind2web_train_sft_history.json", "w"))
