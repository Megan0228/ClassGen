'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
 * References: clip
 * https://github.com/openai/CLIP
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import json
import math
from tqdm import tqdm

import hashlib
import os
import urllib
import warnings
from typing import Any, Union, List

import torch
from tqdm import tqdm

from clip.model import build_model
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from Graph.GCA import GCAModule

_tokenizer = _Tokenizer()

_MODELS = {
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
}


def _download(url: str, root: str = os.path.expanduser("~/.cache/clip")):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True,
                  unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError(f"Model has been downloaded but the SHA256 checksum does not not match")

    return download_target


def available_models() -> List[str]:
    """Returns the names of available CLIP models"""
    return list(_MODELS.keys())


def load(name: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
         jit: bool = False, download_root: str = None, mask: bool = False):
    """Load a CLIP model

    Parameters
    ----------
    name : str
        A model name listed by `clip.available_models()`, or the path to a model checkpoint containing the state_dict

    device : Union[str, torch.device]
        The device to put the loaded model

    jit : bool
        Whether to load the optimized JIT model or more hackable non-JIT model (default).

    download_root: str
        path to download the model files; by default, it uses "~/.cache/clip"

    Returns
    -------
    model : torch.nn.Module
        The CLIP model
    """
    if name in _MODELS:
        model_path = _download(_MODELS[name], download_root or os.path.expanduser("~/.cache/clip"))
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found; available models = {available_models()}")

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location=device if jit else "cpu").eval()
        state_dict = None
    except RuntimeError:
        # loading saved state dict
        if jit:
            warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
            jit = False
        state_dict = torch.load(model_path, map_location="cpu")

    if not jit:
        model = build_model(state_dict or model.state_dict(), mask=mask).to(device)
        if str(device) == "cpu":
            model.float()
        return model

    # patch the device names
    device_holder = torch.jit.trace(lambda: torch.ones([]).to(torch.device(device)), example_inputs=[])
    device_node = [n for n in device_holder.graph.findAllNodes("prim::Constant") if "Device" in repr(n)][-1]

    def patch_device(module):
        try:
            graphs = [module.graph] if hasattr(module, "graph") else []
        except RuntimeError:
            graphs = []

        if hasattr(module, "forward1"):
            graphs.append(module.forward1.graph)

        for graph in graphs:
            for node in graph.findAllNodes("prim::Constant"):
                if "value" in node.attributeNames() and str(node["value"]).startswith("cuda"):
                    node.copyAttributes(device_node)

    model.apply(patch_device)
    patch_device(model.encode_image)
    patch_device(model.encode_text)

    # patch dtype to float32 on CPU
    if str(device) == "cpu":
        float_holder = torch.jit.trace(lambda: torch.ones([]).float(), example_inputs=[])
        float_input = list(float_holder.graph.findNode("aten::to").inputs())[1]
        float_node = float_input.node()

        def patch_float(module):
            try:
                graphs = [module.graph] if hasattr(module, "graph") else []
            except RuntimeError:
                graphs = []

            if hasattr(module, "forward1"):
                graphs.append(module.forward1.graph)

            for graph in graphs:
                for node in graph.findAllNodes("aten::to"):
                    inputs = list(node.inputs())
                    for i in [1, 2]:  # dtype can be the second or third argument to aten::to()
                        if inputs[i].node()["value"] == 5:
                            inputs[i].node().copyAttributes(float_node)

        model.apply(patch_float)
        patch_float(model.encode_image)
        patch_float(model.encode_text)

        model.float()

    return model


def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> torch.LongTensor:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result


class clip_classifier(nn.Module):

    def __init__(self, args, is_train=False):
        super().__init__()
        self.args = args

        self.templates = json.load(open(args.template, 'r'))
        self.templates = self.templates[args.dataset]

        classnames = json.load(open(args.classname, 'r'))
        self.classnames = classnames[args.dataset]

        self.sup_sub_map = json.load(
            open("labels_set/{}/{}-{}.json".format(args.dataset, args.dataset, args.label_set_size), 'r'))
        self.subclassnames = []
        self.sub2sup = {}
        s_id, t_id = [], []
        count = 0
        for id_cls, cls in enumerate(self.classnames):
            for id_subcls, subcls in enumerate(self.sup_sub_map[cls][1:]):
                id_subcls = count + id_subcls
                self.sub2sup[id_subcls] = id_cls  # sub2sup
                s_id.append(len(self.classnames) + id_subcls)
                t_id.append(id_cls)
            self.subclassnames = self.subclassnames + self.sup_sub_map[cls][1:]  # subclassnames
            count = count + len(self.sup_sub_map[cls][1:])
        self.supsub_map = (s_id, t_id)
        self.prob_sc = F.one_hot(torch.tensor(t_id)).float().to(args.device)  # 默认按照targets其中的最大值+1作为one_hot编码的长度

        self.model = load(args.clip_model, jit=False, mask=args.mask)
        self.model.float()
        self.logit_scale = self.model.logit_scale
        self.ctx_dim = self.model.ln_final.weight.shape[0]

        # initialize classifier with class names
        self.init_classifier_weights(args)
        # initialize sub_classifier with subclass names
        self.init_sub_classifier_weights(args)
        # self.init_classifier_weights_add_classinfo(args)

        self.gnn_encoder = GATModel(in_feat=self.ctx_dim, out_feat=self.ctx_dim)

        # delete_unused_modules
        self.delete_unused_modules()
        # self.model = self.model.to(args.device)

    # def load_class_description(self, classnames):
    #     path = "/public/qlp/datasets/class_description"
    #     classinfo_dir = os.path.join(path, classnames+".txt")
    #     class_infos = {}
    #     with open(classinfo_dir) as f:
    #         for line in f:
    #             if line != "\n":
    #                 (key, val) = line.split(":")
    #                 class_infos[key.lower()] = val
    #     return class_infos

    # def count_words(self, text):
    #     words = text.split()
    #     total_words = len(words)
    #     return total_words
    #
    # def process_text(text):
    #     total_words = self.count_words(text)
    #     if total_words > 77:
    #         words = text.split()
    #         truncated_words = words[:max_words]
    #         truncated_text = " ".join(truncated_words)
    #         return truncated_text
    #
    #     else:
    #         return text
    #
    # def init_classifier_weights_add_classinfo(self, args):
    #
    #     print(f"{len(self.classnames)} classes, {len(self.templates)} templates")
    #     classname_infos = self.load_class_description(args.dataset)
    #
    #     with torch.no_grad():
    #         zeroshot_weights = []
    #         for classname in tqdm(self.classnames):
    #             try:
    #                 classname_info = classname_infos[classname.lower()]
    #             except:
    #                 classname_info = classname_infos[classname.lower().replace(" ","")]
    #             texts = [template.format(classname)+classname_info for template in self.templates]  # format with class
    #             texts = tokenize(texts,truncate=True).to(args.device)  # tokenize
    #             class_embeddings = self.model.encode_text(texts)  # embed with text encoder
    #             class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
    #             class_embedding = class_embeddings.mean(dim=0)
    #             class_embedding /= class_embedding.norm()
    #             zeroshot_weights.append(class_embedding)
    #
    #     self.model.visual.classifier = nn.Parameter(torch.stack(zeroshot_weights, dim=1).to(args.device))
    #
    #     # delete unused modules
    #     del self.model.transformer, self.model.token_embedding, self.model.positional_embedding, self.model.ln_final, self.model.text_projection, self.model.logit_scale
    #
    #     return

    def init_classifier_weights(self, args):

        print(f"{len(self.classnames)} classes, {len(self.templates)} templates")

        with torch.no_grad():
            zeroshot_weights = []
            for classname in tqdm(self.classnames):
                texts = [template.format(classname) for template in self.templates]  # format with class
                texts = tokenize(texts).to(args.device)  # tokenize
                class_embeddings = self.model.encode_text(texts)  # embed with text encoder
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)

        self.model.visual.classifier = nn.Parameter(
            torch.stack(zeroshot_weights, dim=1).to(args.device))  # [n_dim, n_cls]

    def init_sub_classifier_weights(self, args):

        print(f"{len(self.subclassnames)} subclasses, {len(self.templates)} templates")

        with torch.no_grad():
            zeroshot_weights = []
            for subclassname in tqdm(self.subclassnames):
                texts = [template.format(subclassname) for template in self.templates]  # format with class
                texts = tokenize(texts).to(args.device)  # tokenize
                class_embeddings = self.model.encode_text(texts)  # embed with text encoder
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)

        self.model.visual.sub_classifier = nn.Parameter(torch.stack(zeroshot_weights, dim=1).to(args.device))

    def delete_unused_modules(self):
        del self.model.transformer, self.model.token_embedding, self.model.positional_embedding, self.model.ln_final, self.model.text_projection, self.model.logit_scale

    def encode_image(self, images, **kwargs):
        return self.model.visual(images, **kwargs)

    def forward(self, images, proto_feat_ori=None, g=None, mask=None):

        n_cluster = proto_feat_ori.shape[0]

        text_feat = self.model.visual.classifier.T
        subcls_feat = self.model.visual.sub_classifier.T

        txt_proto_features = torch.cat([text_feat, subcls_feat, proto_feat_ori], dim=0)
        txt_proto_features = self.gnn_encoder(g, txt_proto_features)
        txt_proto_features = txt_proto_features[:, 0, 0, 0, :].squeeze()
        text_feat = txt_proto_features[: len(self.classnames), :]
        subcls_feat = txt_proto_features[len(self.classnames): len(self.classnames) + len(self.subclassnames), :]
        proto_feat_new = txt_proto_features[len(self.classnames) + len(self.subclassnames):, :]

        if self.args.mask:
            logits, batch_feature, x_mask, x_recon, mask, w, _ = self.encode_image(images, mask=mask)
        else:
            logits, batch_feature, _, _ = self.encode_image(images)

        # 归一化的问题？？
        prob_ic = F.softmax(batch_feature @ text_feat.T, dim=-1)  # [bz, n_cls]
        p1 = prob_ic

        prob_is = F.softmax(batch_feature @ subcls_feat.T, dim=-1)  # [bz, n_subcls]
        p2 = prob_is @ self.prob_sc

        prob_ip = F.softmax(batch_feature @ proto_feat_new.T, dim=-1)  # [bz, n_clusters]
        prob_ps = F.softmax(proto_feat_new @ subcls_feat.T, dim=-1)  # [n_clusters, n_subcls]
        p3 = prob_ip @ prob_ps @ self.prob_sc

        prob = 0.6 * p1 + 0.2 * p2 + 0.2 * p3

        if self.args.mask:
            return prob, proto_feat_new, logits, batch_feature, x_mask, x_recon, mask, w
        else:
            return prob, proto_feat_new, logits, batch_feature


# Our Method
class GCACLIP(nn.Module):
    def __init__(self, args, is_train=False):
        super().__init__()
        self.args = args

        self.templates = json.load(open(args.template, 'r'))
        self.templates = self.templates[args.dataset]

        classnames_allds = json.load(open(args.classname, 'r'))
        self.classnames = classnames_allds[args.dataset]

        self.model = load(args.clip_model, jit=False, mask=args.mask)
        self.model.float()

        # 温度系数不可学习
        self.logit_scale = self.model.logit_scale
        self.logit_scale.requires_grad = False

        self.ctx_dim = self.model.ln_final.weight.shape[0]
        self.dtype = self.model.dtype

        self.base_text_features = self._get_base_text_features()

        self.gnn_encoder = GCAModule(d_model=self.ctx_dim, n_head=self.ctx_dim//64)

        # delete_unused_modules
        self.delete_unused_modules()
        # self.model = self.model.to(args.device)


    def _get_base_text_features(self):
        # if self.model.dtype == torch.float16:
        #     self.model.transformer  = self.model.transformer.cuda()

        print(f"{len(self.classnames)} classes, {len(self.templates)} templates")

        with torch.no_grad():
            text_embeddings = []
            for classname in tqdm(self.classnames):
                texts = [template.format(classname) for template in self.templates]  # format with class
                texts = tokenize(texts).to(self.args.device)  # tokenize
                class_embeddings = self.model.encode_text(texts)  # embed with text encoder
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                text_embeddings.append(class_embedding)

        return torch.stack(text_embeddings, dim=0).to(self.args.device)

    # def setup_parapmeters(self):
    #     #     self.model.eval()
    #     #     self.model.requires_grad_(False)
    #     #     # visual related layer norms
    #     #     for m in self.model.visual.modules():
    #     #         if isinstance(m, torch.nn.LayerNorm) or isinstance(m, torch.nn.BatchNorm2d):
    #     #             m.requires_grad_(True)


    def delete_unused_modules(self):
        del self.model.transformer, self.model.token_embedding, self.model.positional_embedding, self.model.ln_final, self.model.text_projection, self.model.logit_scale

    def encode_image(self, images, **kwargs):
        return self.model.visual(images, **kwargs)

    def forward(self, images, proto_feat_ori=None, g=None, mask=None):
        logit_scale = self.logit_scale.exp()

        if mask is not None:
            batch_feature, x_mask, x_recon, mask, w = self.encode_image(images, mask=mask)
        else:
            batch_feature, _ = self.encode_image(images)

        if proto_feat_ori is None or g is None:
            logits = logit_scale * batch_feature @ self.base_text_features.T
            return logits, batch_feature

        n_cls = len(self.classnames)
        n_cluster = proto_feat_ori.shape[0]
        proto_feat_ori = F.normalize(proto_feat_ori, dim=-1)


        txt_proto_features = torch.cat([self.base_text_features, proto_feat_ori], dim=0)

        # print(txt_proto_features)
        txt_proto_features, _ = self.gnn_encoder(txt_proto_features, g)
        # print(txt_proto_features)


        # text_feat = (1-self.alpha) * self.base_text_features + self.alpha * txt_proto_features[: len(self.classnames), :]
        # proto_feat_new = (1-self.alpha) * proto_feat_ori + self.alpha * txt_proto_features[len(self.classnames): , :]
        self.text_feat = txt_proto_features[: len(self.classnames), :]
        self.proto_feat_new = txt_proto_features[len(self.classnames):, :]

        self.text_feat = F.normalize(self.text_feat, dim=-1)
        self.proto_feat_new = F.normalize(self.proto_feat_new, dim=-1)


        logits = logit_scale * batch_feature @ self.text_feat.T
        if mask is not None:
            return logits, batch_feature, x_mask, x_recon, mask, w, self.proto_feat_new, self.text_feat
        else:
            return logits, batch_feature, self.proto_feat_new, self.text_feat



class CHGCLIP(nn.Module):
    def __init__(self, args, is_train=False):
        super().__init__()
        self.args = args

        self.templates = json.load(open(args.template, 'r'))
        self.templates = self.templates[args.dataset]

        classnames = json.load(open(args.classname, 'r'))
        self.classnames = classnames[args.dataset]

        self.model = load(args.clip_model, jit=False, mask=args.mask)
        self.model.float()
        self.logit_scale = self.model.logit_scale
        self.ctx_dim = self.model.ln_final.weight.shape[0]
        self.dtype = self.model.dtype
        #
        self.base_text_features = self._get_base_text_features()

        self.gnn_encoder = GATModel(in_feat=self.ctx_dim, out_feat=self.ctx_dim)
        # self.alpha = nn.Parameter(torch.ones([]) * 0.5)

        # delete_unused_modules
        self.delete_unused_modules()
        # self.model = self.model.to(args.device)

    def _get_base_text_features(self):
        # if self.model.dtype == torch.float16:
        #     self.model.transformer  = self.model.transformer.cuda()

        print(f"{len(self.classnames)} classes, {len(self.templates)} templates")

        with torch.no_grad():
            text_embeddings = []
            for classname in tqdm(self.classnames):
                texts = [template.format(classname) for template in self.templates]  # format with class
                texts = tokenize(texts).to(self.args.device)  # tokenize
                class_embeddings = self.model.encode_text(texts)  # embed with text encoder
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                text_embeddings.append(class_embedding)

        return torch.stack(text_embeddings, dim=0).to(self.args.device)

    # delete text encoder
    def delete_unused_modules(self):
        del self.model.transformer, self.model.token_embedding, self.model.positional_embedding, self.model.ln_final, self.model.text_projection, self.model.logit_scale

    def encode_image(self, images, **kwargs):
        return self.model.visual(images, **kwargs)

    def forward(self, images, proto_feat_ori=None, g=None, mask=None):
        logit_scale = self.logit_scale.exp()
#
        if mask is not None:
            batch_feature, x_mask, x_recon, mask, w = self.encode_image(images, mask=mask)
        else:
            batch_feature, _ = self.encode_image(images)

        if proto_feat_ori is None or g is None:
            logits = logit_scale * batch_feature @ self.base_text_features.T
            return logits, batch_feature

        n_cls = len(self.classnames)
        n_cluster = proto_feat_ori.shape[0]
        proto_feat_ori = F.normalize(proto_feat_ori, dim=-1)

        self.alpha = 0.9
        txt_proto_features = torch.cat([self.base_text_features, proto_feat_ori], dim=0)
        # print(txt_proto_features)
        txt_proto_features = self.gnn_encoder(g, txt_proto_features)
        txt_proto_features = txt_proto_features[:, 0, 0, 0, :].squeeze()
        # print(txt_proto_features)
        # import pdb;
        # pdb.set_trace()
        text_feat = (1-self.alpha) * self.base_text_features + self.alpha * txt_proto_features[: len(self.classnames), :]
        text_feat = F.normalize(text_feat, dim=-1)
        proto_feat_new = (1-self.alpha) * proto_feat_ori + self.alpha * txt_proto_features[len(self.classnames): , :]
        proto_feat_new = F.normalize(proto_feat_new, dim=-1)
        # print(self.alpha)

        logits = logit_scale * batch_feature @ text_feat.T
        if mask is not None:
            return logits, batch_feature, x_mask, x_recon, mask, w, proto_feat_new, text_feat
        else:
            return logits, batch_feature, proto_feat_new, text_feat




class GATModel(nn.Module):
    def __init__(self, in_feat, hidden_size=1024, out_feat=None):
        super(GATModel, self).__init__()
        self.conv1 = dgl.nn.GATConv(in_feat, hidden_size, num_heads=8)
        self.conv2 = dgl.nn.GATConv(hidden_size, hidden_size, num_heads=8)
        self.conv3 = dgl.nn.GATConv(hidden_size, out_feat, num_heads=8)
        self.gelu = nn.GELU()

    def forward(self, g, node):
        """

        Args:
            g: Directed Graph，dgl.graph对象
            node: The node feature matrix

        Returns:

        """
        h = self.conv1(g, node)
        h = self.gelu(h)
        h = self.conv2(g, h)
        h = self.gelu(h)
        h = self.conv3(g, h)
        return h


class GraphAdapterCLIP(nn.Module):
    def __init__(self, args, is_train=False):
        super().__init__()
        self.args = args

        self.templates = json.load(open(args.template, 'r'))
        self.templates = self.templates[args.dataset]

        classnames = json.load(open(args.classname, 'r'))
        self.classnames = classnames[args.dataset]

        self.model = load(args.clip_model, jit=False, mask=args.mask)
        self.model.float()
        self.logit_scale = self.model.logit_scale
        self.ctx_dim = self.model.ln_final.weight.shape[0]
        self.dtype = self.model.dtype
        #
        self.base_text_features = self._get_base_text_features()

        self.gnn_encoder = GraphLearner(args, classnames, clip_model, self.base_text_features)
        # self.alpha = nn.Parameter(torch.ones([]) * 0.5)
        self.alpha = args.alpha

        # delete_unused_modules
        self.delete_unused_modules()
        # self.model = self.model.to(args.device)

    def _get_base_text_features(self):
        # if self.model.dtype == torch.float16:
        #     self.model.transformer  = self.model.transformer.cuda()

        print(f"{len(self.classnames)} classes, {len(self.templates)} templates")

        with torch.no_grad():
            text_embeddings = []
            for classname in tqdm(self.classnames):
                texts = [template.format(classname) for template in self.templates]  # format with class
                texts = tokenize(texts).to(self.args.device)  # tokenize
                class_embeddings = self.model.encode_text(texts)  # embed with text encoder
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                text_embeddings.append(class_embedding)

        return torch.stack(text_embeddings, dim=0).to(self.args.device)

    # delete text encoder
    def delete_unused_modules(self):
        del self.model.transformer, self.model.token_embedding, self.model.positional_embedding, self.model.ln_final, self.model.text_projection, self.model.logit_scale

    def encode_image(self, images, **kwargs):
        return self.model.visual(images, **kwargs)

    def forward(self, images, proto_feat_ori=None, g=None, mask=None):
        logit_scale = self.logit_scale.exp()

        if mask is not None:
            batch_feature, x_mask, x_recon, mask, w = self.encode_image(images, mask=mask)
        else:
            batch_feature, _ = self.encode_image(images)

        if proto_feat_ori is None or g is None:
            logits = logit_scale * batch_feature @ self.base_text_features.T
            return logits, batch_feature

        n_cls = len(self.classnames)
        n_cluster = proto_feat_ori.shape[0]
        proto_feat_ori = F.normalize(proto_feat_ori, dim=-1)

        # self.alpha = 0.6
        txt_proto_features = torch.cat([self.base_text_features, proto_feat_ori], dim=0)

        # print(txt_proto_features)
        txt_proto_features = self.gnn_encoder(txt_proto_features, g)
        # print(txt_proto_features)
        # import pdb ; pdb.set_trace()

        text_feat = (1-self.alpha) * self.base_text_features + self.alpha * txt_proto_features[: len(self.classnames), :]
        text_feat = F.normalize(text_feat, dim=-1)
        proto_feat_new = (1-self.alpha) * proto_feat_ori + self.alpha * txt_proto_features[len(self.classnames): , :]
        proto_feat_new = F.normalize(proto_feat_new, dim=-1)


        logits = logit_scale * batch_feature @ text_feat.T
        if mask is not None:
            return logits, batch_feature, x_mask, x_recon, mask, w, proto_feat_new, text_feat
        else:
            return logits, batch_feature, proto_feat_new, text_feat


class GraphConvolution(nn.Module):
    def __init__(self, hidden_dim, name=None, device=None, class_num=None, sparse_inputs=False, act=nn.Tanh, bias=True,
                 dropout=0.0):
        super().__init__()
        self.act = nn.Tanh()
        self.device = device
        self.dropout = dropout
        self.sparse_inputs = sparse_inputs
        self.hidden_dim = hidden_dim
        self.bias = bias
        self.hidden_dim = 512
        self.class_num = class_num
        self.gcn_weights = nn.Parameter(torch.ones(self.hidden_dim, self.hidden_dim))
        if self.bias:
            self.gcn_bias = nn.Parameter(torch.zeros(class_num, self.hidden_dim))

        # else:
        #     self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.gcn_weights.size(1))
        self.gcn_weights.data.uniform_(-stdv, stdv)
        # if self.bias is not None:
        #     self.gcn_bias.data.uniform_(-stdv, stdv)

    def forward(self, feat, adj):
        x = feat  # [100, 1024, 101]
        node_size = adj.size()[1]
        adj = torch.clip(adj, min=0.0)
        I = torch.eye(node_size, device='cuda').unsqueeze(dim=0).to(self.device)
        adj = adj + I  # [1000, m+1, m+1]
        adj = graph_norm_ours(adj, batch=True, self_loop=True, symmetric=True)  # [1000, m+1, m+1]
        x = x.transpose(1, 2)
        pre_sup = torch.matmul(x, self.gcn_weights)  # [m+1, 1000, 1024]
        output = torch.matmul(adj, pre_sup)  # [1000, m+1, 1024]

        if self.bias:
            output += self.gcn_bias.unsqueeze(1)
        if self.act is not None:
            return self.act(output[:, 0, :])
        else:
            return output[:, 0, :]

def graph_norm_ours(A, batch=False, self_loop=True, symmetric=True):
    # A = A + I    A: (bs, num_nodes, num_nodes
    # Degree
    d = A.sum(-1)  # (bs, num_nodes) #[1000, m+1]
    if symmetric:
        # D = D^-1/2
        d = torch.pow(d, -0.5)
        if batch:
            D = A.detach().clone()
            for i in range(A.size(0)):
                D[i] = torch.diag(d[i])
            norm_A = D.bmm(A).bmm(D)
        else:
            D = torch.diag(d)
            norm_A = D.mm(A).mm(D)
    else:
        # D=D^-1
        d = torch.pow(d, -1)
        if batch:
            D = A.detach().clone()
            for i in range(A.size(0)):
                D[i] = torch.diag(d[i])
            norm_A = D.bmm(A)
        else:
            D = torch.diag(d)
            norm_A = D.mm(A)

    return norm_A

class GraphLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model, base_text_features, base_img_features):
        super().__init__()
        self.device = clip_model.dtype
        # self.alpha = cfg.TRAINER.COOP.RESIDUAL_SCALE
        self.alpha = 0.1
        print(">> DCT scale factor: ", self.alpha)
        self.register_buffer("base_text_features", base_text_features)  # [1000, 1024]
        self.register_buffer("base_img_features", base_img_features)
        # self.alpha_it = cfg.TRAINER.GRAPHADAPTER.ALPHA
        self.alpha_it = 0.7
        self.beta_it = cfg.TRAINER.GRAPHADAPTER.BETA
        self.node_num = 1
        # self.alpha_it =
        self.hidden_dim = 1
        self.GCN_tt = GraphConvolution(self.hidden_dim, name='metagraph', device=self.device,
                                       class_num=base_text_features.size()[0])
        self.GCN_it = GraphConvolution(self.hidden_dim, name='metagraph', device=self.device,
                                       class_num=base_text_features.size()[0])

    def reset_parameters(self):
        for i in range(self.node_num):
            stdv = 1. / math.sqrt(self.graph_node[i].size(0))
            self.graph_node[i].data.uniform_(-stdv, stdv)

    def forward(self, img_feature):
        sigma = 2.0

        with torch.no_grad():
            node_cluster_t = self.base_text_features.view(1, self.base_text_features.size()[0] // 4, 4,
                                                          self.base_text_features.size()[1])
            node_cluster_i = self.base_img_features.view(1, self.base_img_features.size()[0] // 4, 4,
                                                         self.base_img_features.size()[1])

        graph_o_t_all = []

        for index in range(4):
            # print("========index", index)
            with torch.no_grad():
                inputs_text = self.base_text_features.unsqueeze(dim=1)  # [100, 1, 1024]
                inputs_img = img_feature.unsqueeze(dim=1)
                node_cluster_tt = node_cluster_t[:, :, index, :].repeat(inputs_text.size()[0], 1,
                                                                        1)  # [100, 100, 1024] t->t
                node_cluster_it = node_cluster_i[:, :, index, :].repeat(inputs_text.size()[0], 1, 1)  # i -> t
                feat_tt = torch.cat([inputs_text, node_cluster_tt], dim=1)
                feat_it = torch.cat([inputs_text, node_cluster_it], dim=1)
                feat_tt = feat_tt.transpose(1, 2).detach()
                feat_it = feat_it.transpose(1, 2).detach()
                edge_tt = cal_edge_emb(feat_tt).detach()
                edge_it = cal_edge_emb(feat_it).detach()
            graph_o_tt = self.GCN_tt(feat_tt, edge_tt)
            graph_o_it = self.GCN_it(feat_it, edge_it)
            graph_o_t = (graph_o_tt) * self.alpha_it + (1 - self.alpha_it) * graph_o_it
            graph_o_t_all.append(graph_o_t)
        graph_o_t = torch.stack(graph_o_t_all, dim=0).mean(dim=0)

        return self.beta_it * self.base_text_features + (1 - self.beta_it) * graph_o_t.squeeze(), img_feature

