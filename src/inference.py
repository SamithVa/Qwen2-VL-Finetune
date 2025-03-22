from showui.modeling_showui import ShowUIForConditionalGeneration
from showui.processing_showui import ShowUIProcessor

from qwen_vl_utils import process_vision_info
import time, torch, re

def parse_layer_type(str_ranges, L=28, default=0):
    # 0 is without layer token selection, 1 is with layer token selection. Below we provide examples:
    # [1,28,1] means that all LM layers use token selection; [1,28,0] means that do not.
    # Interleaved layer-wise '[2,2,1],[4,4,1],[6,6,1],[8,8,1],[10,10,1],[12,12,1],[14,14,1],[16,16,1],[18,18,1],[20,20,1],[22,22,1],[24,24,1],[26,26,1]'
    result = [default] * L
    matches = re.findall(r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]', str_ranges)
    for start, end, value in matches:
        start, end, value = int(start) - 1, int(end) - 1, int(value)
        if end >= L:
            end = L - 1
        result[start:end + 1] = [value] * (end - start + 1)
    return result

if __name__=="__main__":
    model_path = "/data/data1/syc/intern/wanshan/models/Qwen2-VL-2B-Instruct"
    device = 'cuda'

    min_pixels = 256*28*28
    max_pixels = 1344*28*28
    # 1. Screenshot -> Graph
    uigraph_train = True        # Enable ui graph during training
    uigraph_test = True         # Enable ui graph during inference
    uigraph_diff = 1            # Pixel difference used for constructing ui graph
    uigraph_rand = False        # Enable random graph construction 
    # 2. Graph -> Mask 
    uimask_pre = True           # Prebuild patch selection mask in the preprocessor (not in model layers) for efficiency
    uimask_ratio = 0.5          # Specify the percentage of patch tokens to skip per component
    uimask_rand = False         # Enable random token selection instead of uniform selection
    lm_skip_ratio = uimask_ratio # valid if not uimask_pre
    lm_skip_layer = "[1,28,1]"   # [1,28,1] means we apply UI guide token selection from 1-th to 28-th layer (28 is the last layer of Qwen2-VL)
    model_path = "/data/data1/syc/intern/wanshan/models/Qwen2-VL-2B-Instruct"
    
    processor = ShowUIProcessor.from_pretrained(
        model_path, 
        min_pixels=min_pixels, max_pixels=max_pixels,
        uigraph_train=uigraph_train, uigraph_test=uigraph_test, uigraph_diff=uigraph_diff, uigraph_rand=uigraph_rand,
        uimask_pre=True, uimask_ratio=uimask_ratio, uimask_rand=uimask_rand,
    )

    messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "./chrome.png",
                "resized_height": 28 * 40,
                "resized_width": 28 * 40,
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)

    print(inputs['select_mask'].shape, inputs['select_mask'].sum())

    lm_qwen_layer = 28 # LLM Decoder layers

    lm_skip_layer = parse_layer_type(lm_skip_layer, lm_qwen_layer)

    model = ShowUIForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        lm_skip_ratio=lm_skip_ratio, lm_skip_layer=lm_skip_layer,
    )

    
    # print(model)
    start_time = time.time()
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    print(generated_ids.shape)
    elapsed_time = time.time() - start_time
    print(elapsed_time)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text)
    # start_time = time.time()
    # elapsed_time = time.time() - start_time
    # print(f'elased_time : {elapsed_time}')