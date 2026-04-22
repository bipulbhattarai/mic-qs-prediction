#!/usr/bin/env python3
"""
Script 01: Fetch MIC-related paper abstracts from PubMed
Paper: Predicting MIC Rates from QS Features using XGBoost-SHAP
Author: Bipul Bhattarai, University of South Dakota

Runs on macOS Apple Silicon (M4) — no special dependencies needed.
Output: data/raw/pubmed_abstracts.jsonl
"""

import requests, json, time, pathlib, sys
from datetime import datetime

OUT_DIR = pathlib.Path("data/raw")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PUBMED_SEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_FETCH  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
PUBMED_SUMM   = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

# Search queries targeting papers with MIC rate measurements + biofilm/QS data
QUERIES = [
    # Core MIC + measurement papers
    "microbiologically influenced corrosion rate measurement biofilm sulfate reducing bacteria",
    "MIC corrosion rate SRB biofilm electrochemical",
    "biocorrosion rate quorum sensing sulfate reducing bacteria",
    "microbiologically induced corrosion rate carbon steel biofilm",
    # QS + corrosion connection
    "quorum sensing biofilm corrosion metal pipeline",
    "quorum sensing SRB Desulfovibrio corrosion rate",
    "AHL autoinducer biofilm corrosion infrastructure",
    # Wastewater + corrosion
    "wastewater biofilm corrosion microbial community",
    "biofouling corrosion rate microbial community measurement",
    # Machine learning + MIC
    "machine learning microbiologically influenced corrosion prediction",
    "XGBoost corrosion prediction microbial",
]

MAX_PER_QUERY = 50  # fetch up to 50 papers per query
DELAY         = 0.4  # seconds between API calls (NCBI rate limit)

def search_pubmed(query: str, max_results: int = 50) -> list[str]:
    """Search PubMed and return list of PMIDs."""
    params = {
        "db"      : "pubmed",
        "term"    : query,
        "retmax"  : max_results,
        "retmode" : "json",
        "sort"    : "relevance",
    }
    resp = requests.get(PUBMED_SEARCH, params=params, timeout=15)
    resp.raise_for_status()
    ids = resp.json().get("esearchresult", {}).get("idlist", [])
    return ids

def fetch_abstracts(pmids: list[str]) -> list[dict]:
    """Fetch title + abstract for a list of PMIDs."""
    if not pmids:
        return []
    params = {
        "db"      : "pubmed",
        "id"      : ",".join(pmids),
        "rettype" : "abstract",
        "retmode" : "xml",
    }
    resp = requests.get(PUBMED_FETCH, params=params, timeout=30)
    resp.raise_for_status()
    xml = resp.text

    # Simple XML parsing — avoid lxml dependency for M4 compatibility
    import re
    articles = []
    article_blocks = re.findall(
        r"<PubmedArticle>(.*?)</PubmedArticle>", xml, re.DOTALL
    )

    for block in article_blocks:
        pmid_match    = re.search(r"<PMID[^>]*>(\d+)</PMID>", block)
        title_match   = re.search(r"<ArticleTitle>(.*?)</ArticleTitle>", block, re.DOTALL)
        abstract_match= re.search(r"<AbstractText[^>]*>(.*?)</AbstractText>", block, re.DOTALL)
        journal_match = re.search(r"<Title>(.*?)</Title>", block)
        year_match    = re.search(r"<PubDate>.*?<Year>(\d{4})</Year>", block, re.DOTALL)

        if pmid_match and (title_match or abstract_match):
            def clean(s):
                return re.sub(r"<[^>]+>", " ", s or "").strip()

            articles.append({
                "pmid"    : pmid_match.group(1),
                "title"   : clean(title_match.group(1) if title_match else ""),
                "abstract": clean(abstract_match.group(1) if abstract_match else ""),
                "journal" : clean(journal_match.group(1) if journal_match else ""),
                "year"    : year_match.group(1) if year_match else "",
            })

    return articles

def main():
    print("=" * 60)
    print("Step 1: Fetching MIC paper abstracts from PubMed")
    print(f"Queries: {len(QUERIES)}")
    print(f"Max per query: {MAX_PER_QUERY}")
    print("=" * 60)

    all_articles = {}  # pmid → article (deduplication)

    for i, query in enumerate(QUERIES, 1):
        print(f"\n[{i}/{len(QUERIES)}] Query: {query[:70]}...")
        try:
            pmids = search_pubmed(query, MAX_PER_QUERY)
            print(f"  Found: {len(pmids)} PMIDs")
            time.sleep(DELAY)

            new_pmids = [p for p in pmids if p not in all_articles]
            if not new_pmids:
                print("  All already fetched — skipping")
                continue

            articles = fetch_abstracts(new_pmids)
            for art in articles:
                all_articles[art["pmid"]] = art

            print(f"  Fetched: {len(articles)} | Total unique: {len(all_articles)}")
            time.sleep(DELAY)

        except Exception as e:
            print(f"  ERROR: {e}")
            time.sleep(2)

    # Save
    out_file = OUT_DIR / "pubmed_abstracts.jsonl"
    with open(out_file, "w") as f:
        for art in all_articles.values():
            f.write(json.dumps(art) + "\n")

    print(f"\n{'='*60}")
    print(f"✅ Done: {len(all_articles)} unique papers saved")
    print(f"   Output: {out_file}")
    print(f"   Size: {out_file.stat().st_size // 1024} KB")

    # Show sample
    sample = list(all_articles.values())[:3]
    print(f"\nSample papers:")
    for s in sample:
        print(f"  PMID {s['pmid']}: {s['title'][:80]}")

if __name__ == "__main__":
    main()
