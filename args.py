import argparse

def get_args():
    parser = argparse.ArgumentParser()

    # 基本参数
    data_root = "../../../../mnt/sda/ytzhu/DEAP/deap_windowed"
    face_video_dir = "../../../../mnt/sda/ytzhu/DEAP/face_video_windows_fixed"
    parser.add_argument('--data_root', type=str, default=data_root)
    parser.add_argument('--face_video_dir', type=str, default=face_video_dir)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--figure_dir', type=str, default='./figures')

    # 模型参数
    parser.add_argument('--contrastive', type=bool, default=False)
    # parser.add_argument('--modalities', type=list, default=['eeg', 'eog', 'emg', 'eda', 'bvp', 'tmp', 'resp', 'face'])
    parser.add_argument('--modalities', type=list, default=['face'])
    parser.add_argument('--missing_rate', type=float, default=0.0)  # 模态缺失率
    parser.add_argument('--output_dim', type=int, default=32)  # 每个模态编码后的维度
    parser.add_argument('--expert_dim', type=int, default=32)
    parser.add_argument('--num_modal_experts', type=int, default=1)
    parser.add_argument('--num_emotion_experts', type=int, default=4)
    parser.add_argument('--emotion_out_dim', type=int, default=4)  # 二分类任务 or 回归任务输出维度
    parser.add_argument('--routing', type=str, default='soft')  
    parser.add_argument('--top_k', type=int, default=4)  # 每个模态选择的专家数
    parser.add_argument('--task', type=str, default='regression')  # 任务类型: 'binary', 'regression', 'quadrant'
    parser.add_argument('--model_name', type=str, default='EmotionPipeline')
    parser.add_argument('--hidden_dim', type=int, default=8)

    # 训练参数
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--amp', action='store_true', default=False)  # 是否使用自动混合精度

    # 分布式参数
    parser.add_argument('--distributed', action='store_true', default=False)  # 是否使用分布式训练
    parser.add_argument("--local_rank", type=int, default=0)   # 下划线
    parser.add_argument("--local-rank", type=int, default=0)   # 中横线
    parser.add_argument("--k_fold", type=int, default=5, help="Number of folds for k-fold cross validation")


    # 其他
    parser.add_argument('--resume', type=str, default=None)  # 路径
    parser.add_argument('--plot_curve', action='store_true')  # 是否绘图

    return parser.parse_args()
