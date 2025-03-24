import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import PostNet, VarianceAdaptor, ConversationalContextEncoder, StyleEncoder, StyleMemoryEnhancementEncoder, LinguisticEncoder, LinguisticMemoryEnhancementEncoder
from utils.tools import get_mask_from_lengths

class CompTransTTS(nn.Module):
    """ CompTransTTS """

    def __init__(self, preprocess_config, model_config, train_config, train_flag):
        super(CompTransTTS, self).__init__()
        self.model_config = model_config
        if model_config["block_type"] == "transformer":
            from .transformers.transformer import TextEncoder, Decoder
        # elif model_config["block_type"] == "lstransformer":
        #     from .transformers.lstransformer import TextEncoder, Decoder
        # elif model_config["block_type"] == "fastformer":
        #     from .transformers.fastformer import TextEncoder, Decoder
        # elif model_config["block_type"] == "conformer":
        #     from .transformers.conformer import TextEncoder, Decoder
        # elif model_config["block_type"] == "reformer":
        #     from .transformers.reformer import TextEncoder, Decoder
        else:
            raise ValueError("Unsupported Block Type: {}".format(model_config["block_type"]))

        self.encoder = TextEncoder(model_config)
        self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config, train_config)
        self.decoder = Decoder(model_config)
        self.mel_linear = nn.Linear(
            model_config["transformer"]["decoder_hidden"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        )
        self.postnet = PostNet()

        self.speaker_emb = self.emotion_emb = None
        if model_config["multi_speaker"]:
            self.embedder_type = preprocess_config["preprocessing"]["speaker_embedder"]
            if self.embedder_type == "none":
                with open(
                    os.path.join(
                        preprocess_config["path"]["preprocessed_path"], "speakers.json"
                    ),
                    "r",
                ) as f:
                    n_speaker = len(json.load(f))
                self.speaker_emb = nn.Embedding(
                    n_speaker,
                    model_config["transformer"]["encoder_hidden"],
                )
            else:
                self.speaker_emb = nn.Linear(
                    model_config["external_speaker_dim"],
                    model_config["transformer"]["encoder_hidden"],
                )
        if model_config["multi_emotion"]:
            with open(
                os.path.join(
                    preprocess_config["path"]["preprocessed_path"], "emotions.json"
                ),
                "r",
            ) as f:
                n_emotion = len(json.load(f))
            self.emotion_emb = nn.Embedding(
                n_emotion,
                model_config["transformer"]["encoder_hidden"],
            )
        self.history_type = model_config["history_encoder"]["type"]

        if self.history_type != "none":
            if self.history_type == "Guo":
                self.context_encoder = ConversationalContextEncoder(preprocess_config, model_config)

        # -------------------------------------------- StyleEncoder, StyleMemoryEnhancementEncoder(cross & intra), LinguisticEncoder, LinguisticMemoryEnhancementEncoder(cross & intra) ----------------------------------
        self.style_flag = train_flag
        self.linguistic_flag = train_flag

        self.style_encoder = StyleEncoder(preprocess_config, model_config)
        self.linguistic_encoder = LinguisticEncoder(preprocess_config, model_config)
        self.cross_style_memory_enhancement_encoder = StyleMemoryEnhancementEncoder(preprocess_config, model_config)
        self.cross_linguistic_memory_enhancement_encoder = LinguisticMemoryEnhancementEncoder(preprocess_config, model_config)
        self.intra_style_memory_enhancement_encoder = StyleMemoryEnhancementEncoder(preprocess_config, model_config)
        self.intra_linguistic_memory_enhancement_encoder = LinguisticMemoryEnhancementEncoder(preprocess_config, model_config)

        # --------------------------------------------  StyleEncoder, StyleMemoryEnhancementEncoder(cross & intra), LinguisticEncoder, LinguisticMemoryEnhancementEncoder(cross & intra) ----------------------------------




    def forward(
        self,
        speakers,
        texts,
        src_lens,
        max_src_len,
        mels=None,
        mel_lens=None,
        max_mel_len=None,
        p_targets=None,
        e_targets=None,
        d_targets=None,
        attn_priors=None,
        spker_embeds=None,
        emotions=None,
        history_info=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
        step=None,
    ):
        style_emb = None
        linguistic_emb = None


        (
            text_embs,
            history_lens,
            history_text_embs,
            history_speakers,
            audio_embs,
            history_audio_embs,
        ) = history_info

        if self.style_flag:
            # (16, 11, 256)
            style_emb = self.style_encoder(history_info[4], history_info[5], history_speakers, speakers)
            # (A2, A3, ..., Acur)
            # (16, 10, 256)
            style_emb = style_emb[:,1:, :]

        # A1， A1-2， A1-3， ...， A1-（C-1）
        # (16, 10, 256)
        cross_style_memory_enhancement_emb = self.cross_style_memory_enhancement_encoder(history_info[5], history_speakers)  # cross
        intra_style_memory_enhancement_emb = self.intra_style_memory_enhancement_encoder(history_info[5], history_speakers)  # intra

        if self.linguistic_flag:
            # (16, 11, 256)
            linguistic_emb = self.linguistic_encoder(history_info[0], history_info[2], history_speakers, speakers)
            # (T2, T3, ..., Tcur)
            # (16, 10, 256)
            linguistic_emb = linguistic_emb[:,1:, :]

        # T1， T1-2， T1-3， ...， T1-（C-1）
        # (16, 10, 256)
        cross_linguistic_memory_enhancement_emb = self.cross_linguistic_memory_enhancement_encoder(history_info[2], history_speakers)  # cross
        intra_linguistic_memory_enhancement_emb = self.intra_linguistic_memory_enhancement_encoder(history_info[2], history_speakers)  # intra



        # cross
        cross_linguistic_enhancement_hidden = cross_linguistic_memory_enhancement_emb[:,-1, :]
        cross_style_memory_enhancement_hidden = cross_style_memory_enhancement_emb[:,-1, :]

        # intra
        intra_linguistic_enhancement_hidden = intra_linguistic_memory_enhancement_emb[:, -1, :]
        intra_style_memory_enhancement_hidden = intra_style_memory_enhancement_emb[:, -1, :]


        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )

        texts, text_embeds = self.encoder(texts, src_masks)

        # Context Encoding
        context_encodings = None
        if self.history_type != "none":
            if self.history_type == "Guo":
                (
                    text_embs,
                    history_lens,
                    history_text_embs,
                    history_speakers,
                    audio_embs,
                    history_audio_embs,
                ) = history_info

                context_encodings = self.context_encoder(
                    text_embs,
                    speakers,
                    history_text_embs,
                    history_speakers,
                    history_lens,
                )

        speaker_embeds = None
        if self.speaker_emb is not None:
            if self.embedder_type == "none":
                speaker_embeds = self.speaker_emb(speakers) # [B, H]
            else:
                assert spker_embeds is not None, "Speaker embedding should not be None"
                speaker_embeds = self.speaker_emb(spker_embeds) # [B, H]

        emotion_embeds = None
        if self.emotion_emb is not None:
            emotion_embeds = self.emotion_emb(emotions)

        (
            output,
            p_targets,
            p_predictions,
            e_targets,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
            attn_outs,
        ) = self.variance_adaptor(
            speaker_embeds,
            emotion_embeds,
            context_encodings,
            texts,
            text_embeds,
            src_lens,
            src_masks,
            mels,
            mel_lens,
            mel_masks,
            max_mel_len,
            p_targets,
            e_targets,
            d_targets,
            attn_priors,
            cross_linguistic_enhancement_hidden,  # cross T 1-(c-1)
            cross_style_memory_enhancement_hidden, # cross S 1-(c-1)
            intra_linguistic_enhancement_hidden, # intra
            intra_style_memory_enhancement_hidden,  # intra
            p_control,
            e_control,
            d_control,
            step,
        )

        output, mel_masks = self.decoder(output, mel_masks)
        output = self.mel_linear(output)

        postnet_output = self.postnet(output) + output

        return (
            output,
            postnet_output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
            attn_outs,
            p_targets,
            e_targets,
        ), (
            style_emb,
            cross_style_memory_enhancement_emb,  # cross
            linguistic_emb,
            cross_linguistic_memory_enhancement_emb, # cross
            intra_style_memory_enhancement_emb,  # intra
            intra_linguistic_memory_enhancement_emb,  # intra
        )
