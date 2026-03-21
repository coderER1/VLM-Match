import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from tqdm import tqdm
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve, average_precision_score
import warnings
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import clip
from peft import LoraConfig, get_peft_model, TaskType
from PIL import Image, ImageFile
import matplotlib.pyplot as plt
import easyocr
import cv2
import torchvision.transforms as T
from torchvision import models
ImageFile.LOAD_TRUNCATED_IMAGES = True

warnings.filterwarnings('ignore')
if not hasattr(torch.compiler, 'is_compiling'):
    torch.compiler.is_compiling = lambda: False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TeacherConfig:
    model_path = "/home/cyp/Mul-modal_MOE1/qw4"
    output_dir = "./qwen4b_teacher_scl_final"
    data_path = "/home/cyp/Mul-modal_MOE1/mm_aug_all_entities_out_fullrunwdc/processed_train_data_pluswdc.pickle"

    batch_size = 16
    learning_rate = 1e-5
    num_epochs = 10
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    min_pixels = 256 * 256
    max_pixels = 512 * 512

    lambda_intra = 0.1
    lambda_inter = 0.1
    margin = 0.2

    exp_dir = "./exp_figs1"
    os.makedirs(exp_dir, exist_ok=True)


clip_model, preprocess = clip.load("ViT-B/32", device=device)

detector = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
detector.eval().to(device)

def select_top_region_frcnn(img: Image.Image, text: str, topk=1, gamma=8,
                            score_thresh=0.25, max_boxes=20, padding=5):
    transform = T.Compose([T.ToTensor()])
    img_tensor = transform(img).to(device)

    with torch.no_grad():
        preds = detector([img_tensor])[0]

    boxes = preds['boxes'][preds['scores'] > score_thresh]
    scores = preds['scores'][preds['scores'] > score_thresh]

    if len(boxes) == 0:
        boxes = torch.tensor([[0, 0, img_tensor.shape[2], img_tensor.shape[1]]], device=device)
        scores = torch.tensor([1.0], device=device)

    if len(boxes) > max_boxes:
        top_idx = scores.topk(max_boxes).indices
        boxes = boxes[top_idx]

    region_features = []
    regions = []

    for box in boxes:
        x1, y1, x2, y2 = box.int()
        x1 = max(x1 - padding, 0)
        y1 = max(y1 - padding, 0)
        x2 = min(x2 + padding, img.width)
        y2 = min(y2 + padding, img.height)
        region = img.crop((int(x1), int(y1), int(x2), int(y2)))
        regions.append(region)

        img_input = preprocess(region).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = clip_model.encode_image(img_input)
            feat /= feat.norm(dim=-1, keepdim=True)
            region_features.append(feat)

    region_features = torch.cat(region_features, dim=0)

    max_length = 77
    if len(text) > max_length:
        text = text[:max_length]

    try:
        text_tokens = clip.tokenize([text], truncate=True).to(device)
    except TypeError:
        tokens = clip._tokenizer.encode(text)
        tokens = tokens[:76] + [49407]
        while len(tokens) < 77:
            tokens.append(0)
        text_tokens = torch.tensor(tokens).unsqueeze(0).to(device)

    with torch.no_grad():
        text_feat = clip_model.encode_text(text_tokens)
        text_feat /= text_feat.norm(dim=-1, keepdim=True)

    sim = (text_feat @ region_features.T).squeeze(0)

    if topk == 1:
        idx = sim.argmax().item()
        selected_box = boxes[idx]
        selected_region = regions[idx]
    else:
        weights = torch.softmax(gamma * sim, dim=0)
        selected_region = torch.zeros_like(region_features[0])
        for i, w in enumerate(weights):
            selected_region += w * region_features[i]
        selected_box = None

    return selected_region, selected_box, regions, boxes, sim.cpu().numpy()

class ImprovedEntityExtractor:
    def __init__(self):
        print("加载多语言 NER 模型: xlm-roberta-base-ner-hrl ...")
        try:
            from transformers import pipeline
            self.ner_pipeline = pipeline(
                "ner",
                model="/home/cyp/Mul-modal_MOE1/ProductsNER8",
                aggregation_strategy="simple",
                device=device
            )
            print("✅ 多语言 NER 模型加载成功")
        except Exception as e:
            print("❌ NER 模型加载失败:", e)
            self.ner_pipeline = None

    def extract_entities_with_ner(self, text):
        if not text or self.ner_pipeline is None:
            return []
        try:
            results = self.ner_pipeline(text)
        except Exception:
            return []
        entities = []
        for r in results:
            label = r.get("entity_group", "")
            word = r.get("word", "").replace("##", "").strip()
            score = r.get("score", 0.0)
            if label in ["Product"] and len(word) > 1:
                entities.append({"word": word, "score": score})
        return entities

    def filter_entities(self, text):
        ner_entities = self.extract_entities_with_ner(text)
        if not ner_entities:
            return ""
        ner_entities.sort(key=lambda x: x["score"], reverse=True)
        return ner_entities[0]["word"][:5]

def extract_entities_from_image(reader, extractor, image):
    try:
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        results = reader.readtext(img_cv)
        ocr_text = " ".join([r[1] for r in results]).strip()
        if not ocr_text:
            return "", ""
        ents = extractor.filter_entities(ocr_text)
        return ents, ocr_text
    except Exception as e:
        print("图片实体提取错误:", e)
        return "", ""
def safe_open_rgb(path, fallback_size=(224, 224)):
    try:
        img = Image.open(path)
        img = img.convert("RGB")
        return img
    except Exception as e:
        print(f"[WARN] bad image: {path} | err={e}")
        return Image.new("RGB", fallback_size, (0, 0, 0))


class CrossEncoderDataset(Dataset):
    def __init__(self, data_dict, processor, config, reader=None, extractor=None,
                 use_ocr=True, use_region=True):
        self.text_pairs = data_dict['text_pairs']
        self.image_pairs = data_dict['image_pairs']
        self.labels = data_dict['labels']
        self.processor = processor
        self.config = config
        self.use_ocr = use_ocr
        self.use_region = use_region
        self.reader = reader
        self.entity_extractor = extractor

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        textA, textB = self.text_pairs[idx]
        imgA_path, imgB_path = self.image_pairs[idx]

        imgA_path = imgA_path.replace("\\", "/")
        imgB_path = imgB_path.replace("\\", "/")
        # imgA_path = imgA_path.replace("images/", "imagesproduct/")
        # imgB_path = imgB_path.replace("images/", "imagesproduct/")
        imgA_pil = safe_open_rgb(imgA_path)
        imgB_pil = safe_open_rgb(imgB_path)

        processed_imgA = imgA_pil
        processed_imgB = imgB_pil

        has_ocrA, has_ocrB = 0, 0
        reg_simA, reg_simB = 0.0, 0.0
        used_regionA, used_regionB = 0, 0

        if self.use_ocr:
            entsA, _ = extract_entities_from_image(self.reader, self.entity_extractor, imgA_pil)
            entsB, _ = extract_entities_from_image(self.reader, self.entity_extractor, imgB_pil)
            has_ocrA = 1 if entsA else 0
            has_ocrB = 1 if entsB else 0
            if entsA:
                textA = f"{textA} [NER]: {entsA}"
            if entsB:
                textB = f"{textB} [NER]: {entsB}"

        if self.use_region:
            regionA, _, _, _, simA = select_top_region_frcnn(imgA_pil, textA)
            regionB, _, _, _, simB = select_top_region_frcnn(imgB_pil, textB)
            reg_simA = float(np.max(simA)) if len(simA) else 0.0
            reg_simB = float(np.max(simB)) if len(simB) else 0.0
            if reg_simA >= 0.25:
                processed_imgA = regionA
                used_regionA = 1
            if reg_simB >= 0.25:
                processed_imgB = regionB
                used_regionB = 1

        label = self.labels[idx]
        textA = textA.rstrip() + " "
        textB = textB.rstrip() + " "
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": processed_imgA},
                {"type": "text", "text": textA},
                {"type": "image", "image": processed_imgB},
                {"type": "text", "text": textB},
                {"type": "text", "text": "Are Entity A and Entity B the same real-world entity?"},
            ],
        }]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding="max_length",
            max_length=1024,
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "pixel_values": inputs["pixel_values"],
            "image_grid_thw": inputs["image_grid_thw"],
            "label": torch.tensor(label, dtype=torch.long),

            "meta_has_ocr_any": torch.tensor(1 if (has_ocrA or has_ocrB) else 0, dtype=torch.long),
            "meta_region_sim_max": torch.tensor(max(reg_simA, reg_simB), dtype=torch.float),
            "meta_used_region_any": torch.tensor(1 if (used_regionA or used_regionB) else 0, dtype=torch.long),
        }

def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])

    pixel_values = torch.cat([item['pixel_values'] for item in batch], dim=0)
    image_grid_thw = torch.cat([item['image_grid_thw'] for item in batch], dim=0)

    meta_has_ocr_any = torch.stack([item["meta_has_ocr_any"] for item in batch])
    meta_region_sim_max = torch.stack([item["meta_region_sim_max"] for item in batch])
    meta_used_region_any = torch.stack([item["meta_used_region_any"] for item in batch])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
        "labels": labels,

        "meta_has_ocr_any": meta_has_ocr_any,
        "meta_region_sim_max": meta_region_sim_max,
        "meta_used_region_any": meta_used_region_any,
    }


import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor
from peft import LoraConfig, TaskType, get_peft_model



class Qwen4BTeacherSCL(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        print(f"Loading Qwen-VL for SCL Training from: {config.model_path}")

        self.base_model = Qwen3VLForConditionalGeneration.from_pretrained(
            config.model_path,
            torch_dtype=torch.bfloat16,
            device_map=config.device,
            output_hidden_states=True
        )

        lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        self.base_model = get_peft_model(self.base_model, lora_config)
        self.base_model.print_trainable_parameters()

        self.processor = AutoProcessor.from_pretrained(config.model_path)
        vocab = self.processor.tokenizer.get_vocab()

        self.vision_start_id = vocab.get("<|vision_start|>", None)
        self.vision_end_id   = vocab.get("<|vision_end|>", None)
        self.im_end_id       = vocab.get("<|im_end|>", None)

        if "<|image_pad|>" in vocab:
            self.img_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        elif "<|vision_start|>" in vocab:
            self.img_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        else:
            self.img_token_id = 151643
        print(f"Detected Image Token ID (fallback): {self.img_token_id}")
        print(f"vision_start_id={self.vision_start_id}, vision_end_id={self.vision_end_id}, im_end_id={self.im_end_id}")

        self.INSTR_TEXT = "Are Entity A and Entity B the same real-world entity?"
        tok = self.processor.tokenizer
        self.instr_patterns = [
            tok.encode(self.INSTR_TEXT, add_special_tokens=False),
            tok.encode(" " + self.INSTR_TEXT, add_special_tokens=False),
            tok.encode("\n" + self.INSTR_TEXT, add_special_tokens=False),
        ]

        hidden_dim = 2560  
        self.hidden_dim = hidden_dim

        self.gate_network = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=-1)
        ).to(torch.bfloat16).to(config.device)

        self.classifier = nn.Linear(hidden_dim, 2).to(torch.bfloat16).to(config.device)

    @torch.no_grad()
    def _find_subseq(self, seq_1d: torch.Tensor, subseq: list, start: int, end: int):
        """
        seq_1d: shape [L], dtype long
        subseq: python list[int]
        return: first index if found, else None
        """
        if subseq is None or len(subseq) == 0:
            return None
        if end - start < len(subseq):
            return None

        sub = torch.tensor(subseq, device=seq_1d.device, dtype=seq_1d.dtype)
        last = end - len(subseq)
        for i in range(start, last + 1):
            if torch.equal(seq_1d[i:i + len(subseq)], sub):
                return i
        return None

    def _locate_instruction_start(self, ids: torch.Tensor, search_start: int, seq_len: int):
        instr_pos = None
        for pat in self.instr_patterns:
            pos = self._find_subseq(ids, pat, start=search_start, end=seq_len)
            if pos is not None:
                instr_pos = pos if instr_pos is None else min(instr_pos, pos)
        return instr_pos

    # ---------- 动态切分：ImageA/TextA/ImageB/TextB ----------
    def _extract_modal_features_dynamic(self, hidden_states, input_ids, attention_mask):

        feats = hidden_states[0]  # embedding layer output
        B, L = input_ids.shape

        V_A_list, T_A_list, V_B_list, T_B_list = [], [], [], []

        for b in range(B):
            seq_len = int(attention_mask[b].sum().item())
            ids = input_ids[b, :seq_len]

            if self.vision_start_id is not None and self.vision_end_id is not None:
                vs = (ids == self.vision_start_id).nonzero(as_tuple=True)[0]
                ve = (ids == self.vision_end_id).nonzero(as_tuple=True)[0]

                if vs.numel() >= 2 and ve.numel() >= 2:
                    vs1, vs2 = int(vs[0].item()), int(vs[1].item())
                    ve1, ve2 = int(ve[0].item()), int(ve[1].item())

                    imgA_start, imgA_end = vs1 + 1, ve1
                    imgB_start, imgB_end = vs2 + 1, ve2

                    txtA_start, txtA_end = ve1 + 1, vs2

                    txtB_start = ve2 + 1
                    instr_pos = self._locate_instruction_start(ids, search_start=txtB_start, seq_len=seq_len)
                    if instr_pos is not None:
                        txtB_end = instr_pos
                    else:
                        if self.im_end_id is not None:
                            im_end_pos = (ids == self.im_end_id).nonzero(as_tuple=True)[0]
                            txtB_end = int(im_end_pos[0].item()) if im_end_pos.numel() > 0 else seq_len
                        else:
                            txtB_end = seq_len

                else:
                    imgA_start = imgA_end = imgB_start = imgB_end = None
                    txtA_start = txtA_end = txtB_start = txtB_end = None
            else:
                imgA_start = imgA_end = imgB_start = imgB_end = None
                txtA_start = txtA_end = txtB_start = txtB_end = None

            if imgA_start is None:
                img_indices = (ids == self.img_token_id).nonzero(as_tuple=True)[0]
                if img_indices.numel() < 2:
                    zero = torch.zeros(feats.shape[-1], device=feats.device, dtype=feats.dtype)
                    V_A_list.append(zero); T_A_list.append(zero)
                    V_B_list.append(zero); T_B_list.append(zero)
                    continue

                diffs = img_indices[1:] - img_indices[:-1]
                break_points = (diffs > 1).nonzero(as_tuple=True)[0]
                if break_points.numel() == 0:
                    zero = torch.zeros(feats.shape[-1], device=feats.device, dtype=feats.dtype)
                    V_A_list.append(zero); T_A_list.append(zero)
                    V_B_list.append(zero); T_B_list.append(zero)
                    continue

                split_idx = int(break_points[0].item())
                idx_img_A_start = int(img_indices[0].item())
                idx_img_A_end   = int(img_indices[split_idx].item())
                idx_img_B_start = int(img_indices[split_idx + 1].item())
                idx_img_B_end   = int(img_indices[-1].item())

                txtA_start, txtA_end = idx_img_A_end + 1, idx_img_B_start
                txtB_start = idx_img_B_end + 1

                instr_pos = self._locate_instruction_start(ids, search_start=txtB_start, seq_len=seq_len)
                if instr_pos is not None:
                    txtB_end = instr_pos
                else:
                    if self.im_end_id is not None:
                        im_end_pos = (ids == self.im_end_id).nonzero(as_tuple=True)[0]
                        txtB_end = int(im_end_pos[0].item()) if im_end_pos.numel() > 0 else seq_len
                    else:
                        txtB_end = seq_len

                imgA_start, imgA_end = idx_img_A_start, idx_img_A_end + 1
                imgB_start, imgB_end = idx_img_B_start, idx_img_B_end + 1

            def mean_or_zero(x):
                if x is None or x.numel() == 0:
                    return torch.zeros(feats.shape[-1], device=feats.device, dtype=feats.dtype)
                if x.shape[0] == 0:
                    return torch.zeros(feats.shape[-1], device=feats.device, dtype=feats.dtype)
                return x.mean(dim=0)

            v_A = mean_or_zero(feats[b, imgA_start:imgA_end])
            v_B = mean_or_zero(feats[b, imgB_start:imgB_end])
            t_A = mean_or_zero(feats[b, txtA_start:txtA_end])
            t_B = mean_or_zero(feats[b, txtB_start:txtB_end])

            V_A_list.append(v_A); T_A_list.append(t_A)
            V_B_list.append(v_B); T_B_list.append(t_B)

            if b == 0 and getattr(self.config, "debug_split", False):
                dec = self.processor.tokenizer.decode
                print("TextA decoded:", dec(ids[txtA_start:txtA_end].tolist(), skip_special_tokens=False))
                print("TextB decoded:", dec(ids[txtB_start:txtB_end].tolist(), skip_special_tokens=False))
                print("Tail decoded :", dec(ids[txtB_end:seq_len].tolist(), skip_special_tokens=False))

        return (torch.stack(V_A_list), torch.stack(T_A_list),
                torch.stack(V_B_list), torch.stack(T_B_list))

    # ---------- losses ----------
    def alignment_loss(self, V, T):
        V_norm = F.normalize(V, p=2, dim=-1)
        T_norm = F.normalize(T, p=2, dim=-1)
        sim = (V_norm * T_norm).sum(dim=-1)
        return (1.0 - sim).mean()

    # ---------- forward ----------
    def forward(self, input_ids, attention_mask, pixel_values, image_grid_thw,
                labels=None, return_gates=False):

        pixel_values = pixel_values.to(self.base_model.dtype)
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            output_hidden_states=True,
            return_dict=True
        )

        # 分类特征：取最后一个有效 token 的 hidden
        last_hidden_state = outputs.hidden_states[-1]
        batch_size = input_ids.shape[0]
        seq_lengths = attention_mask.sum(dim=1) - 1
        fused_feature = last_hidden_state[torch.arange(batch_size, device=self.config.device), seq_lengths]
        logits = self.classifier(fused_feature)

        loss_total = None
        loss_intra = torch.tensor(0.0, device=self.config.device)
        loss_inter = torch.tensor(0.0, device=self.config.device)
        gate_info = None

        if labels is not None:
            # (1) CE
            loss_ce = F.cross_entropy(logits, labels)

            # (2) 动态提取 A/B 的 V/T
            V_A, T_A, V_B, T_B = self._extract_modal_features_dynamic(
                outputs.hidden_states, input_ids, attention_mask
            )

            # (3) intra align
            l_align_A = self.alignment_loss(V_A, T_A)
            l_align_B = self.alignment_loss(V_B, T_B)
            loss_intra = (l_align_A + l_align_B) / 2

            # (4) gate fusion
            combinedA = torch.cat([V_A, T_A], dim=-1)
            modal_weightsA = self.gate_network(combinedA)
            text_weightA = modal_weightsA[:, 0].unsqueeze(-1)
            image_weightA = modal_weightsA[:, 1].unsqueeze(-1)
            fused_featuresA = text_weightA * T_A + image_weightA * V_A

            combinedB = torch.cat([V_B, T_B], dim=-1)
            modal_weightsB = self.gate_network(combinedB)
            text_weightB = modal_weightsB[:, 0].unsqueeze(-1)
            image_weightB = modal_weightsB[:, 1].unsqueeze(-1)
            fused_featuresB = text_weightB * T_B + image_weightB * V_B

            lambda_ent = getattr(self.config, "lambda_ent", 0.005)
            entA = -(modal_weightsA * (modal_weightsA + 1e-8).log()).sum(dim=-1).mean()
            entB = -(modal_weightsB * (modal_weightsB + 1e-8).log()).sum(dim=-1).mean()
            loss_gate = -(entA + entB) / 2  # 负号=最大化熵    

            # (5) contrastive (margin)
            entityA_repr = F.normalize(fused_featuresA, p=2, dim=-1)
            entityB_repr = F.normalize(fused_featuresB, p=2, dim=-1)

            distances = F.pairwise_distance(entityA_repr, entityB_repr, p=2)
            margin = self.config.margin

            # labels: 1 正样本靠近，0 负样本推远+ lambda_ent * loss_gate
            pos_loss = labels * torch.pow(distances, 2)
            neg_loss = (1 - labels) * torch.pow(torch.clamp(margin - distances, min=0.0), 2)
            loss_inter = torch.mean(pos_loss + neg_loss) 

            # total+ self.config.lambda_inter * loss_inter
            loss_total = loss_ce + self.config.lambda_intra * loss_intra  +self.config.lambda_inter * loss_inter

            if return_gates:
                w_img = ((image_weightA + image_weightB) / 2).squeeze(-1).detach().float()
                w_txt = ((text_weightA + text_weightB) / 2).squeeze(-1).detach().float()
                gate_info = {"w_img": w_img, "w_txt": w_txt}

        return loss_total, logits, fused_feature, loss_intra, loss_inter, gate_info



def train_epoch(model, loader, optimizer, device):
    model.train()
    losses, intras, inters = [], [], []
    all_preds, all_labels = [], []
    loop = tqdm(loader, desc="Training (SCL)")

    for batch in loop:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        pixel_values = batch['pixel_values'].to(device)
        image_grid_thw = batch['image_grid_thw'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        loss, logits, _, l_intra, l_inter, _ = model(input_ids, attention_mask, pixel_values, image_grid_thw, labels, return_gates=False)
        loss.backward()
        optimizer.step()

        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.detach().cpu().numpy())

        losses.append(loss.item())
        intras.append(l_intra.item())
        inters.append(l_inter.item())

        loop.set_postfix(L=f"{loss.item():.3f}", In=f"{l_intra.item():.3f}", Out=f"{l_inter.item():.3f}")

    acc = accuracy_score(all_labels, all_preds)
    return float(np.mean(losses)), float(np.mean(intras)), float(np.mean(inters)), float(acc)

@torch.no_grad()
def evaluate_basic(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for batch in tqdm(loader, desc="Evaluating"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        pixel_values = batch['pixel_values'].to(device)
        image_grid_thw = batch['image_grid_thw'].to(device)
        labels = batch['labels'].to(device)

        _, logits, _, _, _, _ = model(input_ids, attention_mask, pixel_values, image_grid_thw, labels, return_gates=False)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    p = precision_score(all_labels, all_preds, zero_division=0)
    r = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    acc = accuracy_score(all_labels, all_preds)
    return float(acc), float(p), float(r), float(f1)

@torch.no_grad()
@torch.no_grad()
def evaluate_probs_and_meta(model, loader, device, return_gates=False):
    model.eval()
    probs, ys = [], []
    ocr_any, reg_sim = [], []
    w_img = []

    for batch in tqdm(loader, desc="Eval (probs/meta)"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        pixel_values = batch['pixel_values'].to(device)
        image_grid_thw = batch['image_grid_thw'].to(device)
        labels = batch['labels'].to(device)

        _, logits, _, _, _, gate = model(
            input_ids, attention_mask, pixel_values, image_grid_thw,
            labels, return_gates=return_gates
        )

        # ---- FIX: bf16 -> fp32 before numpy ----
        prob = torch.softmax(logits.float(), dim=-1)[:, 1]
        probs.append(prob.detach().cpu().numpy())
        ys.append(labels.detach().cpu().numpy())

        ocr_any.append(batch["meta_has_ocr_any"].detach().cpu().numpy())
        reg_sim.append(batch["meta_region_sim_max"].detach().cpu().numpy())

        if return_gates and gate is not None:
            # gate["w_img"] in your model is already float(), but keep consistent
            w_img.append(gate["w_img"].detach().cpu().numpy())

    out = {
        "prob": np.concatenate(probs),
        "y": np.concatenate(ys).astype(int),
        "ocr_any": np.concatenate(ocr_any).astype(int),
        "reg_sim": np.concatenate(reg_sim).astype(float),
    }
    if return_gates and len(w_img) > 0:
        out["w_img"] = np.concatenate(w_img).astype(float)
    return out


def save_fig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

def plot_pr_curve(y_true, prob, out_png):
    precision, recall, _ = precision_recall_curve(y_true, prob)
    ap = average_precision_score(y_true, prob)
    plt.figure(figsize=(5.8, 4.6))
    plt.plot(recall, precision, linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR curve (AUPRC={ap:.4f})")
    save_fig(out_png)


def main():
    conf = TeacherConfig()
    os.makedirs(conf.output_dir, exist_ok=True)
    os.makedirs(conf.exp_dir, exist_ok=True)

    print("Initializing Processor...")
    processor = AutoProcessor.from_pretrained(conf.model_path, min_pixels=conf.min_pixels, max_pixels=conf.max_pixels)

    print(f"Loading data from {conf.data_path}")
    with open(conf.data_path, 'rb') as f:
        train_data, val_data, test_data = pickle.load(f)

    reader = easyocr.Reader(['ch_sim', 'en'], model_storage_directory='EasyOCR')
    entity_extractor = ImprovedEntityExtractor()

    train_ds = CrossEncoderDataset(train_data, processor, conf, reader, entity_extractor, use_ocr=True, use_region=True)
    val_ds   = CrossEncoderDataset(val_data,   processor, conf, reader, entity_extractor, use_ocr=True, use_region=True)
    test_ds  = CrossEncoderDataset(test_data,  processor, conf, reader, entity_extractor, use_ocr=True, use_region=True)

    train_loader = DataLoader(train_ds, batch_size=conf.batch_size, shuffle=True,  collate_fn=collate_fn, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=conf.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=conf.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)

    model = Qwen4BTeacherSCL(conf).to(conf.device)
    optimizer = optim.AdamW(model.parameters(), lr=conf.learning_rate)

    print("Start Training with Dual-SCL...")
    history = []

    for epoch in range(conf.num_epochs):
        print(f"\nEpoch {epoch+1}/{conf.num_epochs}")
        l_total, l_in, l_out, tr_acc = train_epoch(model, train_loader, optimizer, conf.device)
        print(f"Train: Loss={l_total:.4f} (Intra={l_in:.4f}, Inter={l_out:.4f}), Acc={tr_acc:.4f}")

        # 评估阶段代码（根据需要）
        acc, p, r, val_f1 = evaluate_basic(model, val_loader, conf.device)
        print(f"Val: F1={val_f1:.4f}, P={p:.4f}, R={r:.4f}, Acc={acc:.4f}")

        acc, p, r, f1 = evaluate_basic(model, test_loader, conf.device)
        print(f"Test: F1={f1:.4f}, P={p:.4f}, R={r:.4f}, Acc={acc:.4f}")

        history.append({
            "epoch": epoch+1,
            "train_loss": l_total,
            "train_intra": l_in,
            "train_inter": l_out,
            "val_f1": val_f1,
            "test_f1": f1,
        })

    # 保存训练曲线
    with open(os.path.join(conf.exp_dir, "train_curve.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
