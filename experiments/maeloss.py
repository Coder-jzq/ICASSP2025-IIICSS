import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from text import sil_phonemes_ids

class MAELoss(nn.Module):
    """ CompTransTTS Loss """
    def __init__(self, preprocess_config, model_config, train_config):
        super(MAELoss, self).__init__()
        self.loss_config = train_config["loss"]
        self.pitch_feature_level = preprocess_config["preprocessing"]["pitch"][
            "feature"
        ]
        self.energy_feature_level = preprocess_config["preprocessing"]["energy"][
            "feature"
        ]
        self.loss_fn = nn.L1Loss()
        # self.loss_fn = nn.MSELoss()


    def forward(self, inputs, predictions):
        (
            texts,
            _,
            _,
            mel_targets,
            _,
            _,
            pitch_targets,
            energy_targets,
            duration_targets,
            *_,
        ) = inputs[3:]
        (
            mel_predictions,
            postnet_mel_predictions,
            pitch_predictions,
            energy_predictions,
            log_duration_predictions,
            _,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
            attn_outs,
        ) = predictions

        self.src_masks = src_masks = ~src_masks
        mel_masks = ~mel_masks
        mel_targets = mel_targets[:, : mel_masks.shape[1], :]
        self.mel_masks = mel_masks = mel_masks[:, :mel_masks.shape[1]]

        pitch_targets.requires_grad = False
        energy_targets.requires_grad = False
        mel_targets.requires_grad = False

        if self.pitch_feature_level == "phoneme_level":
            pitch_predictions = pitch_predictions.masked_select(src_masks)
            pitch_targets = pitch_targets.masked_select(src_masks)
        elif self.pitch_feature_level == "frame_level":
            pitch_predictions = pitch_predictions.masked_select(mel_masks)
            pitch_targets = pitch_targets.masked_select(mel_masks)

        if self.energy_feature_level == "phoneme_level":
            energy_predictions = energy_predictions.masked_select(src_masks)
            energy_targets = energy_targets.masked_select(src_masks)
        if self.energy_feature_level == "frame_level":
            energy_predictions = energy_predictions.masked_select(mel_masks)
            energy_targets = energy_targets.masked_select(mel_masks)



        mel_predictions = mel_predictions.masked_select(mel_masks.unsqueeze(-1))
        postnet_mel_predictions = postnet_mel_predictions.masked_select(
            mel_masks.unsqueeze(-1)
        )
        mel_targets = mel_targets.masked_select(mel_masks.unsqueeze(-1))

        mel_loss = self.loss_fn(mel_predictions, mel_targets)
        postnet_mel_loss = self.loss_fn(postnet_mel_predictions, mel_targets)

        pitch_loss = self.loss_fn(pitch_predictions, pitch_targets)
        energy_loss = self.loss_fn(energy_predictions, energy_targets)

        attn_soft, attn_hard, attn_hard_dur, attn_logprob = attn_outs
        duration_targets = attn_hard_dur
        nonpadding = self.src_masks.float()

        duration_targets = duration_targets.float() * nonpadding

        duration_targets = (duration_targets + 1).log()

        duration_loss = self.loss_fn(log_duration_predictions, duration_targets)


        return (
            mel_loss,
            postnet_mel_loss,
            pitch_loss,
            energy_loss,
            duration_loss
        )


