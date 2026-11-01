#!/usr/bin/env python3
"""Gera a lista de modelos oficiais gratuitos que nao sao texto puro:
VLMs, embeddings, rerankers, edicao de imagem e 3D.
Uso: python scripts/update_extras.py   (na raiz do repositorio)
"""
import json, csv, os, re, sys, time, urllib.request, urllib.error
from collections import Counter

HF = "https://huggingface.co"
TOKEN = os.environ.get("HF_TOKEN", "").strip()

CREATORS = set("""Qwen stabilityai black-forest-labs Tencent THUDM Wan-AI Lightricks Skywork nvidia
alibaba-pai MiniMaxAI facebook ACE-Step OpenMOSS-Team hexgrad ResembleAI myshell-ai fishaudio nari-labs
ai4bharat coqui microsoft openai ibm-granite pyannote google ByteDance-Seed genmo hpcaitech Kwai-Kolors
PixArt-alpha Alpha-VLLM declare-lab OpenBMB HuggingFaceTB allenai mistralai BAAI jinaai nomic-ai intfloat
Alibaba-NLP mixedbread-ai Snowflake Salesforce CohereLabs CohereForAI meta-llama OpenGVLab zai-org
moonshotai internlm 01-ai LiquidAI""".split())

BAD = re.compile(r"(gguf|ggml|awq|gptq|mlx|4bit|8bit|int4|int8|fp8|fp4|nvfp4|onnx|openvino|lora|"
                 r"controlnet|adapter|ipadapter|vae|upscal|textual|embedding-|quant|merge|uncensored|"
                 r"nsfw|comfy|tiny-random|test|demo|playground)")
FORMAT = re.compile(r"(-diffusers|-bf16|-fp16|-fp32|-safetensors)$")

FREE = {"apache-2.0", "mit", "bsd-3-clause", "bsd-2-clause", "cc0-1.0", "cc-by-4.0", "isc",
        "openrail", "openrail++", "bigcode-openrail-m", "cc-by-sa-4.0", "llama2", "llama3",
        "llama3.1", "llama3.2", "llama3.3", "gemma", "qwen", "other", "cc-by-nc-4.0",
        "cc-by-nc-sa-4.0", "cc-by-nc-nd-4.0", "gpl-3.0", "lgpl-3.0"}
NC = {"cc-by-nc-4.0", "cc-by-nc-sa-4.0", "cc-by-nc-nd-4.0"}
RESTRICTED = {"other", "qwen", "gemma", "llama2", "llama3", "llama3.1", "llama3.2", "llama3.3",
              "openrail", "openrail++", "bigcode-openrail-m"}

CATS = {
    "VLMs (imagem → texto)": (["image-text-to-text"], 10000),
    "Embeddings": (["feature-extraction", "sentence-similarity"], 10000),
    "Rerankers": (["text-ranking"], 1000),
    "Edição de imagem": (["image-to-image"], 5000),
    "3D": (["text-to-3d", "image-to-3d"], 1000),
}

GH_ORG = {
    "Qwen": "QwenLM", "BAAI": "FlagOpen", "jinaai": "jina-ai", "nomic-ai": "nomic-ai",
    "intfloat": "intfloat", "Alibaba-NLP": "Alibaba-NLP", "mixedbread-ai": "mixedbread-ai",
    "Snowflake": "Snowflake-Labs", "Salesforce": "salesforce", "CohereLabs": "CohereLabs",
    "meta-llama": "meta-llama", "OpenGVLab": "OpenGVLab", "zai-org": "zai-org",
    "moonshotai": "MoonshotAI", "internlm": "InternLM", "01-ai": "01-ai", "LiquidAI": "LiquidAI",
    "google": "google-deepmind", "microsoft": "microsoft", "nvidia": "NVIDIA",
    "ibm-granite": "ibm-granite", "stabilityai": "stabilityai",
    "black-forest-labs": "black-forest-labs", "Tencent": "Tencent-Hunyuan",
    "facebook": "facebookresearch", "ByteDance-Seed": "ByteDance-Seed", "OpenBMB": "OpenBMB",
    "THUDM": "THUDM", "allenai": "allenai", "mistralai": "mistralai", "HuggingFaceTB": "huggingface",
    "Lightricks": "Lightricks", "Wan-AI": "Wan-Video",
}

GH_RULES = [
    ("qwen3-vl", "https://github.com/QwenLM/Qwen3-VL"),
    ("qwen3-embedding", "https://github.com/QwenLM/Qwen3-Embedding"),
    ("qwen3-reranker", "https://github.com/QwenLM/Qwen3-Embedding"),
    ("qwen-image-edit", "https://github.com/QwenLM/Qwen-Image"),
    ("bge", "https://github.com/FlagOpen/FlagEmbedding"),
    ("trellis", "https://github.com/microsoft/TRELLIS"),
    ("shap-e", "https://github.com/openai/shap-e"),
    ("hunyuan3d", "https://github.com/Tencent-Hunyuan/Hunyuan3D-2"),
    ("seamless", "https://github.com/facebookresearch/seamless_communication"),
    ("triposr", "https://github.com/VAST-AI-Research/TripoSR"),
    ("flux", "https://github.com/black-forest-labs/flux"),
]


def api(path, tries=4):
    for t in range(tries):
        req = urllib.request.Request(HF + path, headers={"Authorization": "Bearer " + TOKEN} if TOKEN else {})
        try:
            with urllib.request.urlopen(req, timeout=60) as f:
                return json.load(f)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(2 * (t + 1)); continue
            raise
        except Exception:
            time.sleep(2 * (t + 1))
    return []


def github_url(mid):
    name = mid.split("/", 1)[1].lower()
    for pref, url in GH_RULES:
        if name.startswith(pref):
            return url
    gho = GH_ORG.get(mid.split("/")[0])
    return f"https://github.com/{gho}" if gho else ""


def grupo(lic):
    if lic in NC:
        return "Não comercial"
    if lic == "nao-declarada":
        return "Não declarada"
    if lic in RESTRICTED:
        return "Aberta com restrição"
    return "Permissiva"


def existing_ids():
    ids = set()
    for fn in ("modelos.json", "midia.json"):
        if os.path.exists(fn):
            try:
                ids |= {r["id"] for r in json.load(open(fn, encoding="utf-8"))}
            except Exception:
                pass
    return ids


def build():
    skip = existing_ids()
    rows, seen = [], set()
    for cat, (pipes, mindl) in CATS.items():
        got = []
        for pipe in pipes:
            ms = api(f"/api/models?pipeline_tag={pipe}&sort=downloads&direction=-1&limit=500")
            for m in ms if isinstance(ms, list) else []:
                org = m["id"].split("/")[0]
                if org not in CREATORS or m["id"] in skip:
                    continue
                name = m["id"].split("/", 1)[1].lower()
                if BAD.search(name) or FORMAT.search(name):
                    continue
                lic = next((t.split(":", 1)[1] for t in m.get("tags", []) if t.startswith("license:")), "nao-declarada")
                if lic not in FREE and lic != "nao-declarada":
                    continue
                if m.get("downloads", 0) < mindl:
                    continue
                got.append({"id": m["id"], "org": org, "cat": cat, "lic": lic,
                            "dl": m.get("downloads", 0), "likes": m.get("likes", 0), "pipe": pipe})
        seen_fmt = set()
        for r in sorted(got, key=lambda x: -x["dl"]):
            base = FORMAT.sub("", r["id"].split("/", 1)[1].lower())
            if base in seen_fmt or r["id"] in seen:
                continue
            seen_fmt.add(base); seen.add(r["id"]); rows.append(r)
    rows.sort(key=lambda r: -r["dl"])
    return rows


def br(n):
    return f"{n:,}".replace(",", ".")


def write_md(rows):
    cats = Counter(r["cat"] for r in rows)
    L = ["# Outros modelos oficiais e gratuitos\n",
         f"**{len(rows)} modelos** de VLMs, embeddings, rerankers, edição de imagem e 3D, publicados pelas organizações criadoras.\n",
         "Critérios: organização dona do modelo · licença gratuita/aberta · sem quants/LoRAs/merges de terceiros.\n",
         "| Categoria | Modelos |", "|---|---:|"]
    for c in CATS:
        if cats.get(c):
            L.append(f"| {c} | {cats[c]} |")
    L.append("")
    for c in CATS:
        group = [r for r in rows if r["cat"] == c]
        if not group:
            continue
        L += [f"## {c}\n", "| Modelo | Licença | Downloads | GitHub |", "|---|---:|---:|---|"]
        for r in group:
            nome = r["id"].split("/", 1)[1]
            ghu = github_url(r["id"])
            gh = f"[repo]({ghu})" if ghu else "—"
            L.append(f"| [{nome}](https://huggingface.co/{r['id']}) | {r['lic']} | {br(r['dl'])} | {gh} |")
        L.append("")
    L.append("_Gerado automaticamente a partir da API do Hugging Face._")
    open("extras.md", "w", encoding="utf-8").write("\n".join(L) + "\n")


def main():
    print("coletando VLMs, embeddings, rerankers, edição de imagem e 3D...")
    rows = build()
    print("modelos:", len(rows))
    write_md(rows)
    with open("extras.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "organizacao", "categoria", "licenca", "grupo_licenca", "downloads", "likes", "tarefa", "hf_url", "github_url"])
        for r in rows:
            w.writerow([r["id"], r["org"], r["cat"], r["lic"], grupo(r["lic"]), r["dl"], r["likes"],
                        r["pipe"], "https://huggingface.co/" + r["id"], github_url(r["id"])])
    with open("extras.json", "w", encoding="utf-8") as f:
        json.dump([{**r, "grupo_licenca": grupo(r["lic"]), "hf_url": "https://huggingface.co/" + r["id"],
                    "github_url": github_url(r["id"])} for r in rows], f, ensure_ascii=False, indent=1)
    print("arquivos gerados: extras.md, extras.csv, extras.json")


if __name__ == "__main__":
    main()
