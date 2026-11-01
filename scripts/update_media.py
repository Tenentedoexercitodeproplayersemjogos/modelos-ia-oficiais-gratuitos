#!/usr/bin/env python3
"""Gera a lista de modelos NAO-TEXTO oficiais e gratuitos (imagem, video, audio, voz e transcricao).

Uso: python scripts/update_media.py   (na raiz do repositorio)
Variavel opcional: HF_TOKEN
"""
import json, csv, os, re, sys, time, urllib.request, urllib.error
from collections import Counter

HF = "https://huggingface.co"
TOKEN = os.environ.get("HF_TOKEN", "").strip()

# apenas organizacoes donas dos modelos (criadoras)
CREATORS = set("""Qwen stabilityai black-forest-labs Tencent THUDM Wan-AI Lightricks Skywork nvidia
alibaba-pai MiniMaxAI facebook ACE-Step OpenMOSS-Team hexgrad ResembleAI myshell-ai fishaudio nari-labs
ai4bharat coqui microsoft openai ibm-granite pyannote google ByteDance-Seed genmo hpcaitech Kwai-Kolors
PixArt-alpha Alpha-VLLM declare-lab OpenBMB HuggingFaceTB allenai mistralai""".split())

# formatos/derivados que nao entram (quants, LoRAs, ONNX etc.)
BAD = re.compile(r"(gguf|ggml|awq|gptq|mlx|4bit|8bit|int4|int8|fp8|fp4|nvfp4|onnx|openvino|lora|"
                 r"controlnet|adapter|ipadapter|vae|upscal|textual|embedding|quant|merge|uncensored|"
                 r"nsfw|pony|realvis|juggernaut|dreamshaper|tiny-random|test|demo|playground)")
FORMAT = re.compile(r"(-diffusers|-bf16|-fp16|-fp32|-safetensors)$")

FREE = {"apache-2.0", "mit", "bsd-3-clause", "bsd-2-clause", "cc0-1.0", "cc-by-4.0", "isc",
        "openrail", "openrail++", "bigcode-openrail-m", "cc-by-sa-4.0", "llama2", "llama3",
        "llama3.1", "llama3.2", "llama3.3", "gemma", "qwen", "other", "cc-by-nc-4.0",
        "cc-by-nc-sa-4.0", "cc-by-nc-nd-4.0", "gpl-3.0", "lgpl-3.0"}
NC = {"cc-by-nc-4.0", "cc-by-nc-sa-4.0", "cc-by-nc-nd-4.0"}

# categoria -> (pipelines, downloads minimos)
CATS = {
    "Imagem": (["text-to-image"], 1000),
    "Vídeo": (["text-to-video", "image-to-video"], 10000),
    "Áudio/Música": (["text-to-audio"], 5000),
    "Voz (TTS)": (["text-to-speech"], 10000),
    "Transcrição (ASR)": (["automatic-speech-recognition"], 10000),
}

GH_ORG = {
    "stabilityai": "stabilityai", "black-forest-labs": "black-forest-labs", "Qwen": "QwenLM",
    "Tencent": "Tencent-Hunyuan", "THUDM": "THUDM", "Wan-AI": "Wan-Video",
    "Lightricks": "Lightricks", "Skywork": "SkyworkAI", "nvidia": "NVIDIA",
    "alibaba-pai": "alibaba", "MiniMaxAI": "MiniMax-AI", "facebook": "facebookresearch",
    "ACE-Step": "ace-step", "OpenMOSS-Team": "OpenMOSS", "hexgrad": "hexgrad",
    "ResembleAI": "resemble-ai", "myshell-ai": "Myshell-ai", "fishaudio": "fishaudio",
    "nari-labs": "nari-labs", "ai4bharat": "AI4Bharat", "coqui": "coqui-ai",
    "microsoft": "microsoft", "openai": "openai", "ibm-granite": "ibm-granite",
    "pyannote": "pyannote", "google": "google-deepmind", "ByteDance-Seed": "ByteDance-Seed",
    "genmo": "genmoai", "hpcaitech": "hpcaitech", "Kwai-Kolors": "Kwai-Kolors",
    "PixArt-alpha": "PixArt-alpha", "Alpha-VLLM": "Alpha-VLLM", "declare-lab": "declare-lab",
    "OpenBMB": "OpenBMB", "HuggingFaceTB": "huggingface", "allenai": "allenai", "mistralai": "mistralai",
}

GH_RULES = [
    ("wan2.2", "https://github.com/Wan-Video/Wan2.2"),
    ("wan2.1", "https://github.com/Wan-Video/Wan2.1"),
    ("ltx", "https://github.com/Lightricks/LTX-Video"),
    ("hunyuanvideo", "https://github.com/Tencent-Hunyuan/HunyuanVideo"),
    ("hunyuan", "https://github.com/Tencent-Hunyuan"),
    ("cogvideo", "https://github.com/THUDM/CogVideo"),
    ("mochi", "https://github.com/genmoai/mochi"),
    ("open-sora", "https://github.com/hpcaitech/Open-Sora"),
    ("qwen-image", "https://github.com/QwenLM/Qwen-Image"),
    ("flux", "https://github.com/black-forest-labs/flux"),
    ("stable-diffusion", "https://github.com/Stability-AI/generative-models"),
    ("stable-audio", "https://github.com/Stability-AI/stable-audio-tools"),
    ("kokoro", "https://github.com/hexgrad/kokoro"),
    ("xtts", "https://github.com/coqui-ai/TTS"),
    ("whisper", "https://github.com/openai/whisper"),
    ("musicgen", "https://github.com/facebookresearch/audiocraft"),
    ("chatterbox", "https://github.com/resemble-ai/chatterbox"),
    ("dia-", "https://github.com/nari-labs/dia"),
    ("melo", "https://github.com/myshell-ai/MeloTTS"),
    ("fish", "https://github.com/fishaudio/fish-speech"),
    ("vibevoice", "https://github.com/microsoft/VibeVoice"),
    ("pyannote", "https://github.com/pyannote/pyannote-audio"),
    ("pixart", "https://github.com/PixArt-alpha/PixArt-alpha"),
    ("skyreels", "https://github.com/SkyworkAI/SkyReels-V2"),
    ("kolors", "https://github.com/Kwai-Kolors/Kolors"),
    ("lumina", "https://github.com/Alpha-VLLM/Lumina-T2X"),
    ("nemotron", "https://github.com/NVIDIA-NeMo/NeMo"),
    ("granite", "https://github.com/ibm-granite/granite-speech"),
    ("ace-step", "https://github.com/ace-step/ACE-Step"),
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
    return ("Permissiva" if lic in FREE - NC - {"other", "qwen", "gemma", "llama2", "llama3", "llama3.1", "llama3.2", "llama3.3", "openrail", "openrail++"}
            else "Não comercial" if lic in NC
            else "Não declarada" if lic == "nao-declarada"
            else "Aberta com restrição")


def build():
    rows, seen = [], set()
    for cat, (pipes, mindl) in CATS.items():
        got = []
        for pipe in pipes:
            ms = api(f"/api/models?pipeline_tag={pipe}&sort=downloads&direction=-1&limit=500")
            for m in ms if isinstance(ms, list) else []:
                org = m["id"].split("/")[0]
                if org not in CREATORS:
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
        # remove duplicatas de formato (ex.: -Diffusers)
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
    L = ["# Modelos não-texto oficiais e gratuitos\n",
         f"**{len(rows)} modelos** de imagem, vídeo, áudio, voz e transcrição, publicados pelas organizações criadoras.\n",
         "Critérios: organização dona do modelo · licença gratuita/aberta · sem quants/LoRAs/merges de terceiros.\n",
         "A licença `other` indica termos customizados da própria dona (ex.: FLUX.1-dev é gratuita só para uso não comercial).\n",
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
    open("midia.md", "w", encoding="utf-8").write("\n".join(L) + "\n")


def main():
    print("coletando modelos não-texto do Hugging Face...")
    rows = build()
    print("modelos:", len(rows))
    write_md(rows)
    with open("midia.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "organizacao", "categoria", "licenca", "grupo_licenca", "downloads", "likes", "tarefa", "hf_url", "github_url"])
        for r in rows:
            w.writerow([r["id"], r["org"], r["cat"], r["lic"], grupo(r["lic"]), r["dl"], r["likes"],
                        r["pipe"], "https://huggingface.co/" + r["id"], github_url(r["id"])])
    with open("midia.json", "w", encoding="utf-8") as f:
        json.dump([{**r, "grupo_licenca": grupo(r["lic"]), "hf_url": "https://huggingface.co/" + r["id"],
                    "github_url": github_url(r["id"])} for r in rows], f, ensure_ascii=False, indent=1)
    print("arquivos gerados: midia.md, midia.csv, midia.json")


if __name__ == "__main__":
    main()
