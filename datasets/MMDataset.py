from torch.utils.data import Dataset
import torch
import random
import numpy as np

__all__ = ['MMDataset']


class MMDataset(Dataset):

    def __init__(self, label_ids, text_feats, video_feats, audio_feats, drop_rate=0.0):
        self.text_feats  = torch.tensor(text_feats,  dtype=torch.float32)
        self.video_feats = torch.tensor(video_feats, dtype=torch.float32)
        self.audio_feats = torch.tensor(audio_feats, dtype=torch.float32)
        self.label_ids   = torch.tensor(label_ids,   dtype=torch.float32)

        self.drop_rate = drop_rate
        self.size = len(self.text_feats)

    def apply_missing_modality(self, text, audio, video):
        """
        Missing mode mapping:
            0: drop video
            1: drop text
            2: drop audio
            3: drop text + video
            4: drop audio + video
            5: drop text + audio
            6: no drop
        """

        # 不drop
        if random.random() > self.drop_rate:
            return text, audio, video, 6, 0, 0, 0

        missing_mode = random.randint(0, 5)

        missing_text = 0
        missing_audio = 0
        missing_video = 0

        # drop text → text_feats 是 (3,L)
        if missing_mode in [1, 3, 5]:
            text = torch.zeros_like(text)
            missing_text = 1

        # drop audio → (T_a, D_a)
        if missing_mode in [2, 4, 5]:
            audio = torch.zeros_like(audio)
            missing_audio = 1

        # drop video → (T_v, D_v)
        if missing_mode in [0, 3, 4]:
            video = torch.zeros_like(video)
            missing_video = 1

        return text, audio, video, missing_mode, missing_text, missing_audio, missing_video

    def __getitem__(self, index):

        text = self.text_feats[index]    # (3, L)
        video = self.video_feats[index]   # (T_v, D_v)
        audio = self.audio_feats[index]  # (T_a, D_a)
        label = self.label_ids[index]

        text, audio, video, mode, mt, ma, mv = \
            self.apply_missing_modality(text, audio, video)

        return {
            "text_feats": text,
            "video_feats": video,
            "audio_feats": audio,
            "label_ids": label,
            'missing_text': torch.tensor(mt, dtype=torch.bool),
            'missing_audio': torch.tensor(ma, dtype=torch.bool),
            'missing_vision': torch.tensor(mv, dtype=torch.bool),
            'missing_mode': torch.tensor(mode, dtype=torch.long),
        }

    def __len__(self):
        return self.size
