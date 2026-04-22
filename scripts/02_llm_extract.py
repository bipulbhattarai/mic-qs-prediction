#!/usr/bin/env python3
"""
Script 02: Extract structured MIC + QS features from paper abstracts using Claude API
Author: Bipul Bhattarai, University of South Dakota
"""

import json, os, time, pathlib, sys
from tqdm import tqdm

try:
    import anthropic
except ImportError:
    print("ERROR: pip install anthropic")
    sys.exit(1)

IN_FILE  = pathlib.Path("data/raw/pubmed_abstracts.jsonl")
OUT_FILE = pathlib.Path("data/raw/llm_extracted_records.jsonl")
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# Use %s placeholders instead of .format() to avoid JSON brace conflicts
EXTRACTION_PROMPT_TEMPLATE = """You are a bioinformatics data extraction expert.
Extract structured data from the following paper title and abstract about
microbiologically influenced corrosion (MIC) or biofilm corrosion.

Return ONLY a valid JSON object with these exact fields.
Use null for any field not mentioned in the text.
Use numeric values where possible. Do not include units in numeric fields.

{
  "has_mic_rate": true/false,
  "mic_rate_um_per_year": null or number (corrosion rate in micrometers/year; convert if needed),
  "mic_rate_category": null or "low"/"medium"/"high" (<10/10-100/>100 um/yr),
  "corrosion_type": null or "uniform"/"pitting"/"both",
  "metal_type": null or string (e.g. "carbon steel", "stainless steel", "copper"),

  "srb_present": null or true/false,
  "srb_abundance_pct": null or number,
  "iob_present": null or true/false,
  "apb_present": null or true/false,
  "methanogen_present": null or true/false,

  "biofilm_present": null or true/false,
  "biofilm_maturity": null or "early"/"mature"/"dispersal",
  "eps_mentioned": null or true/false,

  "qs_mentioned": null or true/false,
  "ahl_mentioned": null or true/false,
  "qs_inhibitor_used": null or true/false,
  "qs_community_fraction": null or number,

  "temperature_c": null or number,
  "ph": null or number,
  "sulfate_mg_l": null or number,
  "h2s_mg_l": null or number,
  "do_mg_l": null or number,
  "salinity_ppt": null or number,

  "shannon_diversity": null or number,
  "community_size_cells_cm2": null or number,

  "environment": null or "pipeline"/"seawater"/"freshwater"/"wastewater"/"soil"/"lab",
  "study_type": null or "lab"/"field"/"model",
  "exposure_days": null or number,

  "data_usable": true/false
}

Set data_usable=true if at least one of these is present:
mic_rate_um_per_year, metal_type, srb_present, biofilm_present, or qs_mentioned.

PAPER:
Title: TITLE_PLACEHOLDER
Abstract: ABSTRACT_PLACEHOLDER

Return ONLY the JSON object, no other text."""


def build_prompt(title: str, abstract: str) -> str:
    """Build prompt by simple string replacement — avoids JSON brace conflicts."""
    return EXTRACTION_PROMPT_TEMPLATE \
        .replace("TITLE_PLACEHOLDER",    title[:500]) \
        .replace("ABSTRACT_PLACEHOLDER", abstract[:2000])


def extract_with_claude(client, title: str, abstract: str,
                         model: str = "claude-haiku-4-5-20251001") -> dict | None:
    if not abstract and not title:
        return None

    prompt = build_prompt(title, abstract)

    try:
        resp = client.messages.create(
            model      = model,
            max_tokens = 800,
            messages   = [{"role": "user", "content": prompt}]
        )
        text = resp.content[0].text.strip()

        # Strip markdown fences if present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]

        return json.loads(text.strip())

    except json.JSONDecodeError:
        return None
    except Exception as e:
        print(f"\n  API error: {e}")
        return None


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: export ANTHROPIC_API_KEY='sk-ant-...'")
        sys.exit(1)

    if not IN_FILE.exists():
        print(f"ERROR: {IN_FILE} not found. Run 01_fetch_pubmed.py first.")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    papers = []
    with open(IN_FILE) as f:
        for line in f:
            papers.append(json.loads(line.strip()))
    print(f"Loaded {len(papers)} papers")

    # Resume support
    done_pmids = set()
    if OUT_FILE.exists():
        with open(OUT_FILE) as f:
            for line in f:
                try:
                    done_pmids.add(json.loads(line.strip()).get("pmid"))
                except:
                    pass
        if done_pmids:
            print(f"Already processed: {len(done_pmids)} — resuming")

    to_process = [p for p in papers if p["pmid"] not in done_pmids]
    print(f"To process: {len(to_process)} papers")
    print(f"Estimated cost: ~${len(to_process)*0.001:.2f} USD\n")

    usable = 0
    errors = 0

    with open(OUT_FILE, "a") as out_f:
        for paper in tqdm(to_process, desc="Extracting"):
            extracted = extract_with_claude(
                client,
                title    = paper.get("title", ""),
                abstract = paper.get("abstract", "")
            )

            if extracted:
                record = {
                    "pmid"    : paper["pmid"],
                    "title"   : paper.get("title", ""),
                    "journal" : paper.get("journal", ""),
                    "year"    : paper.get("year", ""),
                    **extracted
                }
                out_f.write(json.dumps(record) + "\n")
                out_f.flush()
                if extracted.get("data_usable"):
                    usable += 1
            else:
                errors += 1

            time.sleep(0.3)

    total_out = sum(1 for _ in open(OUT_FILE))
    print(f"\n{'='*55}")
    print(f"✅ Extraction complete")
    print(f"   Total records : {total_out}")
    print(f"   Usable        : {usable} (this run)")
    print(f"   Errors        : {errors}")
    print(f"   Output        : {OUT_FILE}")

    print(f"\nSample usable records:")
    with open(OUT_FILE) as f:
        count = 0
        for line in f:
            rec = json.loads(line)
            if rec.get("data_usable"):
                print(f"  PMID {rec['pmid']}: "
                      f"MIC={rec.get('mic_rate_um_per_year')} um/yr | "
                      f"metal={rec.get('metal_type')} | "
                      f"QS={rec.get('qs_mentioned')} | "
                      f"SRB={rec.get('srb_present')}")
                count += 1
                if count >= 8:
                    break

if __name__ == "__main__":
    main()
