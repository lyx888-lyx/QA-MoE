class Args:
    # general
    dataset = "MIntRec"
    data_path = "../../../data/ytzhu/MIntRec/MIA-datasets/"     # 数据所在目录（audio_feats.pkl、video_feats_fixed.pkl、tsv）
    data_mode = "multi-class"
    logger_name = "Multimodal Intent Recognition"

    # paths inside dataset folder
    text_backbone = "bert-base-uncased"

    # unimodal feature file names
    audio_data_path = "."        # 同目录
    audio_feats_path = "audio_feats.pkl"

    video_data_path = "."
    video_feats_path = "video_feats.pkl"

    # dims and lengths (必须与 benchmark 一致)
    # 但 DataManager 会从 benchmarks 自动读取
    # 所以你不用手动填
    padding_mode = "zero"
    padding_loc = "end"

    # dataloader
    train_batch_size = 16
    eval_batch_size = 16
    num_workers = 0       # Mac/Linux 可以设 4；为了避免死锁先设 0

    # optional
    seed = 42


# for running
args = Args()

from base import DataManager

dm = DataManager(args)

print("Train size:", len(dm.mm_data["train"]))
print("Dev size:", len(dm.mm_data["dev"]))
print("Test size:", len(dm.mm_data["test"]))

# inspect one item
batch = next(iter(dm.mm_dataloader["train"]))
print(batch.keys())
