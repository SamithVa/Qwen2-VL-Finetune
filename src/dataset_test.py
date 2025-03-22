from showui.processing_showui import ShowUIProcessor
from training.params import DataArguments
from training.data import SupervisedDataset, DataCollatorForSupervisedDataset
from torch.utils.data import DataLoader


def make_supervised_data_module(model_id, processor, data_args):
    """Make dataset and collator for supervised fine-tuning."""
    sft_dataset = SupervisedDataset(
        data_path=data_args.data_path,
        processor=processor,
        data_args=data_args,
        model_id=model_id,
    )
    data_collator = DataCollatorForSupervisedDataset(
        pad_token_id=processor.tokenizer.pad_token_id,
        padding_side=processor.tokenizer.padding_side,
        uimask_pre=processor.uimask_pre,
    )

    # Use DataLoader to handle batching
    dataloader = DataLoader(
        sft_dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=data_collator,  # This applies the collator to batches
    )

    # Fetch one batch
    batch = next(iter(dataloader))

    print("Collated Batch Output:")
    for key, value in batch.items():
        print(f"{key}: {value.shape if hasattr(value, 'shape') else value}")

    return dict(
        train_dataset=sft_dataset, eval_dataset=None, data_collator=data_collator, dataloader=dataloader
    )

    # return dict(
    #     train_dataset=sft_dataset, eval_dataset=None, data_collator=data_collator
    # )


if __name__ == "__main__":
    model_path = "/data/data1/syc/intern/wanshan/models/Qwen2-VL-2B-Instruct"
    data_path = "/home/syc/intern/wanshan/Qwen2VL-UI-Graph-Finetune/data/mind2web_train_sft_history.json"

    min_pixels = 256 * 28 * 28
    max_pixels = 1344 * 28 * 28
    # 1. Screenshot -> Graph
    uigraph_train = True  # Enable ui graph during training
    uigraph_test = False  # Enable ui graph during inference
    uigraph_diff = 1  # Pixel difference used for constructing ui graph
    uigraph_rand = False  # Enable random graph construction
    # 2. Graph -> Mask
    uimask_pre = True  # Prebuild patch selection mask in the preprocessor (not in model layers) for efficiency
    uimask_ratio = 0.5  # Specify the percentage of patch tokens to skip per component
    uimask_rand = False  # Enable random token selection instead of uniform selection

    processor = ShowUIProcessor.from_pretrained(
        model_path,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        uigraph_train=uigraph_train,
        uigraph_test=uigraph_test,
        uigraph_diff=uigraph_diff,
        uigraph_rand=uigraph_rand,
        uimask_pre=uimask_pre,
        uimask_ratio=uimask_ratio,
        uimask_rand=uimask_rand,
    )

    model_id = model_path
    data_args = DataArguments(
        data_path,
        image_folder="/data/data1/syc/intern/wanshan/mind2map_dataset/mind2web_images",
    )

    data_module = make_supervised_data_module(model_path, processor, data_args)
    # print(**data_module)
    # print(data_module)
