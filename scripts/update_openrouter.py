#!/usr/bin/env python3
"""Lista os modelos com custo zero na OpenRouter (grátis para usar).

Importante: "grátis para usar" NAO significa licenca livre. A licenca do modelo
(e quando existe) vem do card do Hugging Face, quando ha um hf_id.

Uso: python scripts/update_openrouter.py
"""
import json, csv, os, time, urllib.request, urllib.error

HF = "https://huggingface.co"
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
OR = "https://openrouter.ai"


def get(url, headers=None, tries=4):
    for t in range(tries):
        req = urllib.request.Request(url, headers=headers or {"User-Agent": "opencode"})
        try:
            with urllib.request.urlopen(req, timeout=60) as f:
                return json.load(f)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(2 * (t + 1)); continue
            return None
        except Exception:
            time.sleep(2 * (t + 1))
    return None


def tem_preco(pr):
    """True se qualquer campo de preco for > 0 (inclui overrides)."""
    def num(v):
        try:
            return float(v)
        except Exception:
            return 0.0
    for k, v in (pr or {}).items():
        if k == "overrides":
            continue
        if num(v) > 0:
            return True
    for ov in (pr or {}).get("overrides") or []:
        for k, v in ov.items():
            if k.startswith("min_") or k.startswith("max_"):
                continue
            if num(v) > 0:
                return True
    return False


def hf_info(hf_id):
    if not hf_id:
        return {}
    h = {"Authorization": "Bearer " + HF_TOKEN} if HF_TOKEN else {"User-Agent": "opencode"}
    return get(HF + "/api/models/" + hf_id, headers=h) or {}


def main():
    data = get(OR + "/api/v1/models")
    if not data or "data" not in data:
        print("erro: nao consegui ler a API da OpenRouter", flush=True)
        raise SystemExit(1)

    rows = []
    for m in data["data"]:
        mid = m.get("id", "")
        # apenas os marcados :free e sem nenhum preco > 0
        if not mid.endswith(":free") or tem_preco(m.get("pricing") or {}):
            continue
        org = mid.split("/")[0]
        hf_id = (m.get("hugging_face_id") or "").strip()
        info = hf_info(hf_id)
        lic = next((t.split(":", 1)[1] for t in info.get("tags", []) if t.startswith("license:")), "nao-informada")
        arch = m.get("architecture") or {}
        criado = m.get("created")
        if isinstance(criado, (int, float)):
            criado = time.strftime("%Y-%m-%d", time.gmtime(criado))
        else:
            criado = str(criado or "")[:10]
        rows.append({
            "id": mid,
            "nome": m.get("name") or mid,
            "org": org,
            "ctx": m.get("context_length") or 0,
            "modalidade": arch.get("modality") or "",
            "hf_id": hf_id,
            "hf_url": (HF + "/" + hf_id) if hf_id else "",
            "licenca": lic,
            "dl_hf": info.get("downloads", 0),
            "criado": criado,
            "url": OR + "/" + mid,
        })
    rows.sort(key=lambda r: r["criado"], reverse=True)

    with open("openrouter.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    with open("openrouter.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "nome", "organizacao", "contexto", "modalidade", "hf_id", "licenca_hf",
                    "downloads_hf", "criado", "openrouter_url", "hf_url"])
        for r in rows:
            w.writerow([r["id"], r["nome"], r["org"], r["ctx"], r["modalidade"], r["hf_id"],
                        r["licenca"], r["dl_hf"], r["criado"], r["url"], r["hf_url"]])

    L = ["# Grátis para usar (OpenRouter)\n",
         f"**{len(rows)} modelos** com custo zero na OpenRouter.\n",
         "⚠️ **Grátis para usar não é licença livre.** Aqui o custo é zero, mas os termos de uso são da plataforma e podem mudar. A coluna Licença mostra o que o card do Hugging Face declara (quando existe).\n",
         "| Modelo | Contexto | Modalidade | Licença (HF) | OpenRouter |",
         "|---|---:|---|---|---|"]
    for r in rows:
        L.append(f"| {r['nome']} | {r['ctx']:,} | {r['modalidade']} | {r['licenca']} | [usar]({r['url']}) |".replace(",", "."))
    L.append("\n_Gerado automaticamente a partir da API da OpenRouter._")
    open("openrouter.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("modelos gratis para usar:", len(rows))
    print("arquivos gerados: openrouter.md, openrouter.csv, openrouter.json")


if __name__ == "__main__":
    main()
