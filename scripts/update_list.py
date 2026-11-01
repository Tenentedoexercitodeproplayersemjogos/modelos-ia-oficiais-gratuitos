#!/usr/bin/env python3
"""Gera a lista de modelos de IA oficiais e gratuitos a partir da API do Hugging Face.

Uso: python scripts/update_list.py   (na raiz do repositorio)
Variavel opcional: HF_TOKEN (aumenta o limite de requisicoes da API do HF)
"""
import json, csv, os, re, sys, time, urllib.request, urllib.error
from collections import Counter, defaultdict

HF = "https://huggingface.co"
TOKEN = os.environ.get("HF_TOKEN", "").strip()

ORGS = """Qwen deepseek-ai meta-llama google mistralai openai zai-org moonshotai MiniMaxAI
ibm-granite bigcode microsoft nvidia JetBrains 01-ai THUDM baichuan-inc internlm upstage
CohereForAI tiiuae allenai EleutherAI HuggingFaceTB stabilityai xai-org tencent inclusionAI
OpenGVLab Salesforce nomic-ai jinaai codefuse-ai BAAI Snowflake Databricks ServiceNow
LGAI-EXAONE aisingapore""".split()

NEW_ORGS = """baidu stepfun-ai ByteDance-Seed SarvamAI inceptionai openbmb AI21Labs NousResearch
arcee-ai LiquidAI rekaai Motif-Technologies Tesslate agentica-org""".split()

CODE = re.compile(r"(^|[/_.-])(code|coder|coding|codestral|codellama|codegemma|starcoder|devstral|mellum|codebert|codet5|codegen|unixcoder|graphcodebert|opencode)([/_.-]|$)|nextcoder|wavecoder|aescode|codereviewer")
QUANT = re.compile(r"(gguf|ggml|awq|gptq|mlx|4bit|8bit|int4|int8|fp8|fp4|nvfp4|mxfp4|bnb|exl2|w4a16|quant|-keras|-pytorch|dflash|eagle3)")
JUNK = re.compile(r"(math|guard|shieldgemma|wildguard|t5gemma|diffusion|llada|-vl-|vl-|whisper|ocr|audio|vision|llama-2|pythia|gpt-neo|gpt-j|gpt2|dialogpt|phi-1|phi-2|no-code|no-math-code|datadecide)")
BRAND = {
    "Qwen": "qwen", "deepseek-ai": "deepseek", "ibm-granite": "granite", "bigcode": "starcoder",
    "microsoft": "phi|codebert|unixcoder|wavecoder|nextcoder|aescode", "google": "gemma",
    "meta-llama": "llama", "mistralai": "mistral|devstral|codestral", "zai-org": "glm|codegeex",
    "moonshotai": "kimi", "MiniMaxAI": "minimax", "JetBrains": "mellum", "01-ai": "yi",
    "tiiuae": "falcon", "nvidia": "nemotron|nv-embed|opencode|ising", "allenai": "olmo",
    "EleutherAI": "pythia|gpt", "HuggingFaceTB": "smollm", "stabilityai": "stable",
    "tencent": "hunyuan", "inclusionAI": "ling", "OpenGVLab": "internvl",
    "Salesforce": "codet5|codegen|sfr|xlam", "nomic-ai": "nomic", "jinaai": "jina",
    "codefuse-ai": "f2llm|codefuse", "BAAI": "bge", "Snowflake": "arctic",
    "aisingapore": "sea-lion|sealion",
}

GH_ORG = {
    "Qwen": "QwenLM", "Salesforce": "salesforce", "ibm-granite": "ibm-granite", "nvidia": "NVIDIA",
    "deepseek-ai": "deepseek-ai", "bigcode": "bigcode-project", "meta-llama": "meta-llama",
    "microsoft": "microsoft", "JetBrains": "JetBrains", "EleutherAI": "EleutherAI",
    "google": "google-deepmind", "zai-org": "zai-org", "moonshotai": "MoonshotAI",
    "MiniMaxAI": "MiniMax-AI", "mistralai": "mistralai", "openai": "openai", "01-ai": "01-ai",
    "internlm": "InternLM", "upstage": "upstage", "tiiuae": "tiiuae", "allenai": "allenai",
    "HuggingFaceTB": "huggingface", "stabilityai": "stabilityai", "tencent": "Tencent-Hunyuan",
    "inclusionAI": "inclusionAI", "OpenGVLab": "OpenGVLab", "nomic-ai": "nomic-ai",
    "jinaai": "jina-ai", "codefuse-ai": "codefuse-ai", "BAAI": "FlagOpen",
    "Snowflake": "Snowflake-Labs", "aisingapore": "aisingapore", "baidu": "PaddlePaddle",
    "stepfun-ai": "stepfun-ai", "ByteDance-Seed": "ByteDance-Seed", "SarvamAI": "sarvamai",
    "inceptionai": "inceptionai", "openbmb": "OpenBMB", "NousResearch": "NousResearch",
    "arcee-ai": "arcee-ai", "LiquidAI": "LiquidAI", "rekaai": "RekaAI", "AI21Labs": "AI21Labs",
    "agentica-org": "agentica-org", "CohereForAI": "CohereForAI", "Databricks": "databricks",
    "ServiceNow": "ServiceNow", "LGAI-EXAONE": "LGAI-EXAONE",
}

# repos verificados; ordem importa (mais especifico primeiro)
GH_RULES = [
    ("qwen3-coder", "https://github.com/QwenLM/Qwen3-Coder"),
    ("qwen2.5-coder", "https://github.com/QwenLM/Qwen2.5-Coder"),
    ("qwen3", "https://github.com/QwenLM/Qwen3"),
    ("qwen2.5", "https://github.com/QwenLM/Qwen2.5"),
    ("deepseek-coder", "https://github.com/deepseek-ai/DeepSeek-Coder-V2"),
    ("deepseek-r1", "https://github.com/deepseek-ai/DeepSeek-R1"),
    ("deepseek-v3.2", "https://github.com/deepseek-ai/DeepSeek-V3.2-Exp"),
    ("deepseek-v3", "https://github.com/deepseek-ai/DeepSeek-V3"),
    ("codellama", "https://github.com/meta-llama/codellama"),
    ("llama", "https://github.com/meta-llama/llama-models"),
    ("codegemma", "https://github.com/google-deepmind/gemma"),
    ("gemma", "https://github.com/google-deepmind/gemma"),
    ("starcoder", "https://github.com/bigcode-project/starcoder2"),
    ("granite", "https://github.com/ibm-granite/granite-code-models"),
    ("gpt-oss", "https://github.com/openai/gpt-oss"),
    ("kimi", "https://github.com/MoonshotAI/Kimi-K2"),
    ("minimax", "https://github.com/MiniMax-AI/MiniMax-M2"),
    ("glm", "https://github.com/zai-org/GLM-4.5"),
    ("nemotron", "https://github.com/NVIDIA-NeMo/Nemotron"),
    ("mellum", "https://github.com/JetBrains/Mellum"),
    ("olmo", "https://github.com/allenai/OLMo"),
    ("pythia", "https://github.com/EleutherAI/gpt-neox"),
    ("gpt-neo", "https://github.com/EleutherAI/gpt-neox"),
    ("smollm", "https://github.com/huggingface/smollm"),
    ("codet5", "https://github.com/salesforce/CodeT5"),
    ("codebert", "https://github.com/microsoft/CodeBERT"),
    ("unixcoder", "https://github.com/microsoft/CodeBERT"),
    ("seed-coder", "https://github.com/ByteDance-Seed/Seed-Coder"),
    ("yi-", "https://github.com/01-ai/Yi"),
    ("internlm", "https://github.com/InternLM/InternLM"),
    ("ling-", "https://github.com/inclusionAI/Ling"),
    ("sea-lion", "https://github.com/aisingapore/sea-lion"),
    ("bge-", "https://github.com/FlagOpen/FlagEmbedding"),
]

QUANT_OK = re.compile(r"(gguf|awq|gptq|mlx|4bit|8bit|int4|int8|fp8|fp4|nvfp4|mxfp4|bnb|exl2|w4a16|quant)")

PERM = {"apache-2.0", "mit", "bsd-3-clause", "bsd-2-clause", "cc0-1.0", "cc-by-4.0", "isc"}
NC = {"cc-by-nc-4.0", "cc-by-nc-sa-4.0", "cc-by-nc-nd-4.0"}


def api(path, tries=4):
    for t in range(tries):
        req = urllib.request.Request(HF + path, headers={"Authorization": "Bearer " + TOKEN} if TOKEN else {})
        try:
            with urllib.request.urlopen(req, timeout=45) as f:
                return json.load(f), f.headers.get("Link", "")
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(2 * (t + 1)); continue
            raise
        except Exception:
            time.sleep(2 * (t + 1))
    return [], ""


def list_org(org, cap=6):
    url = f"/api/models?author={org}&limit=1000&sort=downloads&direction=-1"
    out, pages = [], 0
    while url and pages < cap:
        ms, link = api(url)
        if not isinstance(ms, list):
            break
        out += ms; pages += 1
        if 'rel="next"' not in link:
            break
        url = link.split("<")[1].split(">")[0].replace(HF, "")
        time.sleep(0.1)
    return out


def row(m):
    cd = m.get("cardData") or {}
    lic = cd.get("license") or next((t.split(":")[1] for t in m.get("tags", []) if t.startswith("license:")), "nao-declarada")
    return {"id": m["id"], "org": m["id"].split("/")[0], "lic": lic,
            "dl": m.get("downloads", 0), "likes": m.get("likes", 0),
            "pipe": m.get("pipeline_tag", "")}


def grupo(lic):
    return "Permissiva" if lic in PERM else "Não comercial" if lic in NC else "Não declarada" if lic == "nao-declarada" else "Aberta com restrição"


def github_url(mid):
    name = mid.split("/", 1)[1].lower()
    for pref, url in GH_RULES:
        if name.startswith(pref):
            return url
    gho = GH_ORG.get(mid.split("/")[0])
    return f"https://github.com/{gho}" if gho else ""


def load_existing():
    """Carrega a lista ja publicada para que a atualizacao seja incremental."""
    if not os.path.exists("modelos.json"):
        return {}
    try:
        data = {}
        for r in json.load(open("modelos.json", encoding="utf-8")):
            data[r["id"]] = {
                "id": r["id"],
                "org": r.get("org") or r["id"].split("/")[0],
                "lic": r.get("lic") or r.get("licenca") or "nao-declarada",
                "dl": r.get("dl", r.get("downloads", 0)),
                "likes": r.get("likes", 0),
                "pipe": r.get("pipe", r.get("tarefa", "")),
            }
        return data
    except Exception as e:
        print("aviso: nao consegui ler modelos.json:", e, file=sys.stderr)
        return {}


def build():
    data = load_existing()
    print("base existente:", len(data))

    def ver(m):
        """Atualiza metadados de quem ja esta na lista; devolve True se deve pular filtros."""
        mid = m["id"]
        if mid in data:
            d = data[mid]
            d["dl"] = m.get("downloads", d.get("dl", 0))
            d["likes"] = m.get("likes", d.get("likes", 0))
            d["pipe"] = m.get("pipeline_tag") or d.get("pipe", "")
            cd = m.get("cardData") or {}
            lic = cd.get("license") or next((t.split(":")[1] for t in m.get("tags", []) if t.startswith("license:")), None)
            if lic:
                d["lic"] = lic
            return True
        return False

    # 1) especializados em codigo, sem quants, orgs principais
    for org in ORGS:
        for m in list_org(org, cap=2):
            if ver(m):
                continue
            name = m["id"].lower().split("/", 1)[1]
            if QUANT.search(name) or JUNK.search(name) or not CODE.search(name):
                continue
            data[m["id"]] = row(m)
    # 2) generalistas oficiais (>=10k downloads)
    for org in ORGS:
        for m in list_org(org, cap=2):
            if ver(m):
                continue
            name = m["id"].lower().split("/", 1)[1]
            if QUANT.search(name) or JUNK.search(name):
                continue
            if m.get("pipeline_tag") != "text-generation" or m.get("downloads", 0) < 10000:
                continue
            data[m["id"]] = row(m)
    # 3) quants publicados pela propria dona
    for org in ORGS:
        for m in list_org(org, cap=2):
            if ver(m):
                continue
            name = m["id"].lower().split("/", 1)[1]
            if not QUANT_OK.search(name) or not CODE.search(name):
                continue
            own = BRAND.get(org)
            if own and not re.search(own, name):
                continue
            data[m["id"]] = row(m)
    # 4) novas orgs criadoras
    for org in NEW_ORGS:
        for m in list_org(org, cap=2):
            if ver(m):
                continue
            name = m["id"].lower().split("/", 1)[1]
            if QUANT.search(name) or JUNK.search(name) or not CODE.search(name):
                continue
            data[m["id"]] = row(m)
    rows = sorted(data.values(), key=lambda r: -r["dl"])
    return rows


def br(n):
    return f"{n:,}".replace(",", ".")


def write_readme(rows, path="README.md"):
    posorg = defaultdict(list)
    for r in rows:
        posorg[r["org"]].append(r)
    esp = [r for r in rows if CODE.search(r["id"].lower().split("/", 1)[1])]
    gen = [r for r in rows if not CODE.search(r["id"].lower().split("/", 1)[1])]
    g = Counter(grupo(r["lic"]) for r in rows)
    L = []
    A = L.append
    A("# Modelos de IA oficiais e gratuitos\n")
    A(f"Lista de **{len(rows)} modelos** de IA publicados pelas **próprias organizações criadoras**, com licença gratuita/aberta. Inclui modelos especializados em código e generalistas usados para programar.\n")
    A("🔗 **Coleção no Hugging Face:** https://huggingface.co/collections/Darkzinhoo/modelos-de-ia-oficiais-e-gratuitos-6ae598997b5be2d991c1bc10\n")
    A("🌐 **Site com busca e filtros:** https://tenentedoexercitodeproplayersemjogos.github.io/modelos-ia-oficiais-gratuitos/\n")
    A("## Resumo\n")
    A(f"- **{len(rows)}** modelos de **{len(posorg)}** organizações")
    A(f"- **{len(esp)}** especializados em código · **{len(gen)}** generalistas")
    A(f"- **{g['Permissiva']}** licenças permissivas (Apache/MIT/BSD/CC-BY) · **{g['Aberta com restrição']}** abertas com restrição (Llama/Gemma/Qwen/OpenRAIL) · **{g['Não comercial']}** não comerciais · **{g['Não declarada']}** sem licença declarada\n")
    A("## Critérios\n")
    A("- Repositório publicado pela organização dona do modelo (sem quants/fine-tunes/merges de terceiros)")
    A("- Licença gratuita ou aberta (Apache-2.0, MIT, BSD, CC-BY, Llama, Gemma, Qwen, OpenRAIL, CC-BY-NC)")
    A("- Quants oficiais incluídos apenas quando publicados pela própria dona")
    A("- Dados: [`modelos.csv`](modelos.csv) · [`modelos.json`](modelos.json)\n")
    midia = []
    if os.path.exists("midia.json"):
        try:
            midia = json.load(open("midia.json", encoding="utf-8"))
        except Exception:
            midia = []
    if midia:
        cats = Counter(r["cat"] for r in midia)
        A("## Modelos não-texto (imagem, vídeo, áudio e voz)\n")
        A(f"**{len(midia)} modelos** oficiais em {len(cats)} categorias: " +
          " · ".join(f"{k} ({v})" for k, v in cats.items()) + ".\n")
        A("📄 Lista completa: [`midia.md`](midia.md) · [`midia.csv`](midia.csv) · [`midia.json`](midia.json)\n")
    extras = []
    if os.path.exists("extras.json"):
        try:
            extras = json.load(open("extras.json", encoding="utf-8"))
        except Exception:
            extras = []
    if extras:
        cats = Counter(r["cat"] for r in extras)
        A("## Outros modelos (VLMs, embeddings, rerankers, edição e 3D)\n")
        A(f"**{len(extras)} modelos** oficiais em {len(cats)} categorias: " +
          " · ".join(f"{k} ({v})" for k, v in cats.items()) + ".\n")
        A("📄 Lista completa: [`extras.md`](extras.md) · [`extras.csv`](extras.csv) · [`extras.json`](extras.json)\n")
    A("## Índice de organizações\n")
    for org in sorted(posorg, key=lambda o: (-len(posorg[o]), o)):
        A(f"- [{org}](#{org.lower().replace('.', '').replace('_', '-')}) ({len(posorg[org])})")
    A("")
    A("## Modelos por organização\n")
    for org in sorted(posorg, key=lambda o: (-len(posorg[o]), o)):
        A(f"### {org}\n")
        A("| Modelo | Licença | Downloads | GitHub |")
        A("|---|---:|---:|---|")
        for r in sorted(posorg[org], key=lambda x: -x["dl"]):
            nome = r["id"].split("/", 1)[1]
            ghu = github_url(r["id"])
            gh = f"[repo]({ghu})" if ghu else "—"
            A(f"| [{nome}](https://huggingface.co/{r['id']}) | {r['lic']} | {br(r['dl'])} | {gh} |")
        A("")
    A("## Licenças\n")
    A("| Grupo | Licenças |")
    A("|---|---|")
    A("| Permissiva | Apache-2.0, MIT, BSD-2/3, CC0, CC-BY-4.0, ISC |")
    A("| Aberta com restrição | Llama 2/3, Gemma, Qwen, OpenRAIL, BigCode OpenRAIL-M, DeepSeek, outras |")
    A("| Não comercial | CC-BY-NC 4.0, CC-BY-NC-SA 4.0 |")
    A("| Não declarada | card do modelo sem tag de licença |")
    A("")
    A("_Gerado automaticamente a partir da API do Hugging Face. Downloads no momento da coleta._\n")
    open(path, "w", encoding="utf-8").write("\n".join(L))


def main():
    print("coletando modelos do Hugging Face...")
    rows = build()
    print("modelos:", len(rows))
    write_readme(rows)
    with open("modelos.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "organizacao", "licenca", "grupo_licenca", "downloads", "likes", "tarefa", "hf_url", "github_url"])
        for r in rows:
            w.writerow([r["id"], r["org"], r["lic"], grupo(r["lic"]), r["dl"], r["likes"],
                        r.get("pipe", ""), "https://huggingface.co/" + r["id"], github_url(r["id"])])
    with open("modelos.json", "w", encoding="utf-8") as f:
        json.dump([{**r, "grupo_licenca": grupo(r["lic"]),
                    "hf_url": "https://huggingface.co/" + r["id"],
                    "github_url": github_url(r["id"])} for r in rows], f, ensure_ascii=False, indent=1)
    print("arquivos gerados")


if __name__ == "__main__":
    main()
