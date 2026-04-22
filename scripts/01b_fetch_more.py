#!/usr/bin/env python3
"""Fetch additional MIC papers focused on those with reported rate measurements."""

import requests, json, time, pathlib

OUT_DIR  = pathlib.Path("data/raw")
OUT_FILE = OUT_DIR / "pubmed_abstracts.jsonl"

PUBMED_SEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_FETCH  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

# More specific queries targeting papers that report actual rate numbers
EXTRA_QUERIES = [
    "microbiologically influenced corrosion rate mm/year measurement carbon steel",
    "SRB sulfate reducing bacteria corrosion rate micrometer measurement steel",
    "Desulfovibrio corrosion rate measurement electrochemical steel",
    "biofilm corrosion rate weight loss measurement steel pipeline",
    "MIC corrosion rate measurement quorum sensing Pseudomonas aeruginosa",
    "biocorrosion rate measurement sulfate reducing bacteria Desulfovibrio vulgaris",
    "microbiological corrosion pitting rate measurement stainless steel seawater",
    "biofilm metal corrosion electrochemical impedance rate measurement",
    "sulfate reducing bacteria corrosion rate mpy mils per year measurement",
    "MIC rate prediction machine learning biofilm microbial",
    "quorum sensing corrosion metal biofilm AHL acyl homoserine lactone",
    "biofilm community corrosion carbon steel anaerobic SRB rate",
    "biocorrosion copper seawater rate measurement biofilm",
    "microbially influenced corrosion pipeline field measurement rate",
    "corrosion rate measurement biofilm SRB APB iron oxidizing bacteria",
]

def search_pubmed(query, max_results=60):
    resp = requests.get(PUBMED_SEARCH, params={
        "db": "pubmed", "term": query,
        "retmax": max_results, "retmode": "json", "sort": "relevance"
    }, timeout=15)
    resp.raise_for_status()
    return resp.json().get("esearchresult", {}).get("idlist", [])

def fetch_abstracts(pmids):
    if not pmids:
        return []
    import re
    resp = requests.get(PUBMED_FETCH, params={
        "db": "pubmed", "id": ",".join(pmids),
        "rettype": "abstract", "retmode": "xml"
    }, timeout=30)
    resp.raise_for_status()
    xml = resp.text
    articles = []
    for block in re.findall(r"<PubmedArticle>(.*?)</PubmedArticle>", xml, re.DOTALL):
        pmid_m    = re.search(r"<PMID[^>]*>(\d+)</PMID>", block)
        title_m   = re.search(r"<ArticleTitle>(.*?)</ArticleTitle>", block, re.DOTALL)
        abstract_m= re.search(r"<AbstractText[^>]*>(.*?)</AbstractText>", block, re.DOTALL)
        journal_m = re.search(r"<Title>(.*?)</Title>", block)
        year_m    = re.search(r"<PubDate>.*?<Year>(\d{4})</Year>", block, re.DOTALL)
        clean = lambda s: re.sub(r"<[^>]+>", " ", s or "").strip()
        if pmid_m:
            articles.append({
                "pmid"    : pmid_m.group(1),
                "title"   : clean(title_m.group(1) if title_m else ""),
                "abstract": clean(abstract_m.group(1) if abstract_m else ""),
                "journal" : clean(journal_m.group(1) if journal_m else ""),
                "year"    : year_m.group(1) if year_m else "",
            })
    return articles

# Load existing PMIDs
existing = set()
if OUT_FILE.exists():
    with open(OUT_FILE) as f:
        for line in f:
            try: existing.add(json.loads(line)["pmid"])
            except: pass
print(f"Existing papers: {len(existing)}")

all_new = {}
for i, query in enumerate(EXTRA_QUERIES, 1):
    print(f"[{i}/{len(EXTRA_QUERIES)}] {query[:65]}...")
    try:
        pmids = search_pubmed(query, 60)
        new_pmids = [p for p in pmids if p not in existing and p not in all_new]
        if not new_pmids:
            print(f"  No new PMIDs")
            continue
        arts = fetch_abstracts(new_pmids[:30])
        for a in arts:
            all_new[a["pmid"]] = a
        print(f"  New: {len(arts)} | Running total new: {len(all_new)}")
        time.sleep(0.4)
    except Exception as e:
        print(f"  ERROR: {e}")
        time.sleep(2)

# Append new papers
with open(OUT_FILE, "a") as f:
    for art in all_new.values():
        f.write(json.dumps(art) + "\n")

total = sum(1 for _ in open(OUT_FILE))
print(f"\n✅ Added {len(all_new)} new papers")
print(f"   Total in file: {total}")
