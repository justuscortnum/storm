"""Run the STORM Wiki pipeline with GWDG/SAIA models (OpenAI-compatible) and Tavily search.

Prerequisites:
  - secrets.toml with GWDG_API_KEY and TAVILY_API_KEY
  - pip install -r requirements.txt (in the active venv)

Example:
  python run_storm_gwdg.py --output-dir ./results --retriever tavily \
      --context "Focus on evaluation metrics" \
      --do-research --do-generate-outline --do-generate-article --do-polish-article
"""

import os
import re
from argparse import ArgumentParser

import requests
from bs4 import BeautifulSoup

from knowledge_storm import (
    STORMWikiRunnerArguments,
    STORMWikiRunner,
    STORMWikiLMConfigs,
)
from knowledge_storm.lm import LitellmModel
from knowledge_storm.rm import TavilySearchRM, DuckDuckGoSearchRM
from knowledge_storm.utils import load_api_key
from knowledge_storm.storm_wiki.modules import persona_generator as _persona_generator

# GWDG/SAIA endpoint and models. Use instruct (non-reasoning) models only.
GWDG_API_BASE = "https://chat-ai.academiccloud.de/v1"
FAST_MODEL = "openai/openai-gpt-oss-120b"
STRONG_MODEL = "openai/openai-gpt-oss-120b"

# Domains excluded from search results (non-citable sources).
DENY_DOMAINS = (
    "youtube.com", "youtu.be", "quora.com", "reddit.com", "facebook.com",
    "tiktok.com", "pinterest.", "geeksforgeeks.org", "medium.com",
    "linkedin.com", "liner.com", "consuledge", "debuggercafe.com",
    "ijettjournal", "substack.com", "aitinkerers.org",
)

_WIKI_HEADERS = {
    "User-Agent": "STORM-eval/1.0 (academic research; mailto:you@example.com)"
}


def _patched_get_wiki_page_title_and_toc(url):
    """Fetch a Wikipedia page's title and table of contents.

    Adds a User-Agent (Wikipedia blocks requests without one), a timeout,
    a status check, and a guard against a missing <h1>.
    """
    resp = requests.get(url, headers=_WIKI_HEADERS, timeout=20)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.content, "html.parser")
    h1 = soup.find("h1")
    if h1 is None:
        raise ValueError(f"No <h1> found on {url}")
    main_title = h1.text.replace("[edit]", "").strip().replace("\xa0", " ")

    toc, levels = "", []
    excluded = {"Contents", "See also", "Notes", "References", "External links"}
    for header in soup.find_all(["h2", "h3", "h4", "h5", "h6"]):
        level = int(header.name[1])
        section_title = header.text.replace("[edit]", "").strip().replace("\xa0", " ")
        if section_title in excluded:
            continue
        while levels and level <= levels[-1]:
            levels.pop()
        levels.append(level)
        toc += "  " * (len(levels) - 1) + section_title + "\n"
    return main_title, toc.strip()


_persona_generator.get_wiki_page_title_and_toc = _patched_get_wiki_page_title_and_toc


def is_valid_source(url: str) -> bool:
    u = (url or "").lower()
    return not any(d in u for d in DENY_DOMAINS)


def sanitize_topic(topic: str) -> str:
    """Strip characters that are invalid in Windows file/dir names and cap length."""
    cleaned = re.sub(r'[<>:"/\\|?*]', "", topic).strip()
    if len(cleaned) > 200:
        cleaned = cleaned[:200].rsplit(" ", 1)[0]
    return cleaned


def build_runner(args):
    load_api_key(toml_file_path="secrets.toml")

    gwdg_kwargs = {
        "api_key": os.getenv("GWDG_API_KEY"),
        "api_base": GWDG_API_BASE,
        "temperature": 1.0,
        "top_p": 0.9,
        "num_retries": 6,
        "timeout": 120,
    }

    lm_configs = STORMWikiLMConfigs()
    lm_configs.set_conv_simulator_lm(LitellmModel(model=FAST_MODEL, max_tokens=2000, **gwdg_kwargs))
    lm_configs.set_question_asker_lm(LitellmModel(model=FAST_MODEL, max_tokens=2000, **gwdg_kwargs))
    lm_configs.set_outline_gen_lm(LitellmModel(model=STRONG_MODEL, max_tokens=700, **gwdg_kwargs))
    lm_configs.set_article_gen_lm(LitellmModel(model=STRONG_MODEL, max_tokens=1500, **gwdg_kwargs))
    lm_configs.set_article_polish_lm(LitellmModel(model=STRONG_MODEL, max_tokens=4000, **gwdg_kwargs))

    engine_args = STORMWikiRunnerArguments(
        output_dir=args.output_dir,
        max_conv_turn=args.max_conv_turn,
        max_perspective=args.max_perspective,
        search_top_k=args.search_top_k,
        max_thread_num=args.max_thread_num,
    )

    if args.retriever == "tavily":
        rm = TavilySearchRM(
            tavily_search_api_key=os.getenv("TAVILY_API_KEY"),
            k=engine_args.search_top_k,
            is_valid_source=is_valid_source,
            include_raw_content=True,
        )
    elif args.retriever == "duckduckgo":
        rm = DuckDuckGoSearchRM(
            k=engine_args.search_top_k,
            is_valid_source=is_valid_source,
            safe_search="On",
            region="de-de",
        )
    else:
        raise ValueError(f"Unknown retriever: {args.retriever}")

    # Guard Tavily against empty or over-long queries (some models emit these),
    # which would otherwise raise and abort the whole run.
    if hasattr(rm, "tavily_client"):
        _original_search = rm.tavily_client.search

        def _safe_search(query, *a, **k):
            q = str(query).strip() if query else ""
            if not q:
                return {"results": []}
            if len(q) > 400:
                q = q[:400]
            return _original_search(q, *a, **k)

        rm.tavily_client.search = _safe_search

    return STORMWikiRunner(engine_args, lm_configs, rm)


def main(args):
    runner = build_runner(args)

    topic = sanitize_topic(input("Topic: "))
    context = args.context or input("Context/focus (empty for none): ").strip()
    print(f"[info] Topic: {topic}")
    print(f"[info] Context: {context or '(none)'}")

    runner.run(
        topic=topic,
        context=context,
        do_research=args.do_research,
        do_generate_outline=args.do_generate_outline,
        do_generate_article=args.do_generate_article,
        do_polish_article=args.do_polish_article,
    )
    runner.post_run()
    runner.summary()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="./results")
    parser.add_argument("--retriever", type=str, default="tavily",
                        choices=["tavily", "duckduckgo"])
    parser.add_argument("--context", type=str, default="",
                        help="Additional focus/instructions passed to STORM.")
    parser.add_argument("--max-thread-num", type=int, default=1,
                        help="Keep at 1 to avoid GWDG rate limits.")
    parser.add_argument("--max-conv-turn", type=int, default=4)
    parser.add_argument("--max-perspective", type=int, default=4)
    parser.add_argument("--search-top-k", type=int, default=5)
    parser.add_argument("--retrieve-top-k", type=int, default=3)
    parser.add_argument("--do-research", action="store_true")
    parser.add_argument("--do-generate-outline", action="store_true")
    parser.add_argument("--do-generate-article", action="store_true")
    parser.add_argument("--do-polish-article", action="store_true")
    main(parser.parse_args())
