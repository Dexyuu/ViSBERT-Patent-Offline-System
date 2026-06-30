import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import re
from transformers import AutoModel, ViTModel
from torchvision import transforms

def mean_pooling(model_output, attention_mask):
    """Averages the text token embeddings, ignoring padding tokens."""
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class MultimodalEncoder(nn.Module):
    def __init__(self, text_model_name="sentence-transformers/all-MiniLM-L6-v2", proj_dim=256, dropout=0.1):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.image_encoder = ViTModel.from_pretrained("google/vit-base-patch16-224", add_pooling_layer=False)
        self.dropout = nn.Dropout(dropout)
        self.text_proj = nn.Linear(self.text_encoder.config.hidden_size, proj_dim)
        self.image_proj = nn.Linear(self.image_encoder.config.hidden_size, proj_dim)
        self.fusion_proj = nn.Linear(proj_dim * 2, proj_dim)
        self.log_temp = nn.Parameter(torch.tensor(np.log(0.07), dtype=torch.float))

    def _encode_text(self, input_ids, attention_mask):
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        mean_pooled = mean_pooling(text_outputs, attention_mask)
        return F.normalize(self.text_proj(self.dropout(mean_pooled)), p=2, dim=1)

    def _encode_image(self, images):
        image_outputs = self.image_encoder(images)
        mean_pooled = image_outputs.last_hidden_state.mean(dim=1)
        return F.normalize(self.image_proj(self.dropout(mean_pooled)), p=2, dim=1)

    def encode_fused(self, input_ids, attention_mask, images):
        text_proj = self._encode_text(input_ids, attention_mask)
        image_proj = self._encode_image(images)
        fused = torch.cat([text_proj, image_proj], dim=1)
        return F.normalize(self.fusion_proj(self.dropout(fused)), p=2, dim=1)

    def encode_text_as_fused(self, input_ids, attention_mask):
        text_proj = self._encode_text(input_ids, attention_mask)
        zeros = torch.zeros_like(text_proj)
        fused_input = torch.cat([text_proj, zeros], dim=1)
        return F.normalize(self.fusion_proj(self.dropout(fused_input)), p=2, dim=1)

    def encode_image_as_fused(self, images):
        image_proj = self._encode_image(images)
        zeros = torch.zeros_like(image_proj)
        fused_input = torch.cat([zeros, image_proj], dim=1)
        return F.normalize(self.fusion_proj(self.dropout(fused_input)), p=2, dim=1)

    def preprocess_image(self, image):
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return preprocess(image).unsqueeze(0)

    def preprocess_text(self, title, abstract, claims, tokenizer, stop_words):
        def clean_text(text):
            text = text.lower()
            text = re.sub(r"[^a-z0-9\s]", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
            tokens = [word for word in text.split() if word not in stop_words]
            return " ".join(tokens)

        title = clean_text(title or "")
        abstract = clean_text(abstract or "")
        claims = clean_text(claims or "")
        combined_text = f"{title} {abstract} {claims}".strip()

        return tokenizer(
            combined_text,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )