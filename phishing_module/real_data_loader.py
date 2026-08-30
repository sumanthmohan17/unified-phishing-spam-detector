"""
Real-World Phishing & Legitimate Benchmark Dataset Loader
=========================================================
Module 1 Data Loader: Sourcing real phishing URLs from the UCI PhiUSIIL
dataset and augmented realistic complex-path legitimate URLs from top
Tranco/Alexa domains (e-commerce, developer repos, docs, SaaS, news, AI),
extracting live features for empirical ensemble model training as described
in Section 6.3 of the project report.
"""

from __future__ import annotations

import io
import logging
import time
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from .classifier import build_feature_vector
from .feature_extraction import extract_url_features

logger = logging.getLogger("phishing_module.data_loader")

CACHE_DIR = Path(__file__).parent / "data"
CACHE_FILE = CACHE_DIR / "phiusiil_urls.parquet"
FEATURE_CACHE_FILE = CACHE_DIR / "augmented_extracted_features.parquet"

# 100 Curated Realistic Complex-Path Legitimate URLs across major web site categories
COMPLEX_LEGITIMATE_URLS: List[str] = [
    # 1. Developer / Code Repositories & Package Registries (20 URLs)
    "https://github.com/microsoft/vscode/blob/main/README.md",
    "https://github.com/torvalds/linux/tree/master/kernel",
    "https://github.com/facebook/react/releases/tag/v18.2.0",
    "https://github.com/python/cpython/pull/12345/commits",
    "https://github.com/golang/go/issues/54321",
    "https://gitlab.com/gitlab-org/gitlab/-/merge_requests/98765",
    "https://gitlab.com/gitlab-org/gitlab-runner/-/blob/main/Makefile",
    "https://bitbucket.org/atlassian/python-stash/src/master/setup.py",
    "https://npmjs.com/package/express/v/4.18.2",
    "https://npmjs.com/package/@angular/core",
    "https://pypi.org/project/scikit-learn/#history",
    "https://pypi.org/project/xgboost/#files",
    "https://crates.io/crates/tokio/1.35.0",
    "https://hub.docker.com/_/alpine/tags?page=1&name=3.19",
    "https://hub.docker.com/r/library/postgres/tags",
    "https://gist.github.com/octocat/6cad326836d38bd3a7ae",
    "https://raw.githubusercontent.com/psf/requests/main/requests/models.py",
    "https://sourceforge.net/projects/sevenzip/files/7-Zip/23.01/",
    "https://developer.android.com/reference/android/app/Activity",
    "https://developer.apple.com/documentation/swift/array",

    # 2. E-Commerce & Retail Product Pages (20 URLs)
    "https://www.amazon.com/dp/B08N5WRWNW/ref=syn_sd_onsite_desktop_0",
    "https://www.amazon.com/Apple-MacBook-13-inch-256GB-Storage/dp/B08N5N6RSS",
    "https://www.amazon.com/gp/bestsellers/electronics/ref=zg_bs_nav_0",
    "https://www.ebay.com/itm/123456789012?hash=item1c2d3e4f5a:g:abcAAOSw",
    "https://www.ebay.com/sch/i.html?_nkw=laptop+computers&_sacat=0",
    "https://www.walmart.com/ip/Sony-PlayStation-5-Video-Game-Console/123456789",
    "https://www.walmart.com/browse/electronics/smartphones/3944_1234",
    "https://www.target.com/p/apple-airpods-pro-2nd-generation/-/A-86734567",
    "https://www.target.com/c/household-essentials/-/N-5xsz1",
    "https://www.bestbuy.com/site/apple-watch-series-9-gps-45mm/6543210.p?skuId=6543210",
    "https://www.bestbuy.com/site/computer-cards-components/video-graphics-cards/abcat0507002.c",
    "https://www.etsy.com/listing/123456789/handmade-leather-wallet-personalized",
    "https://store.steampowered.com/app/1091500/Cyberpunk_2077/",
    "https://store.steampowered.com/search/?filter=popularnew",
    "https://www.ikea.com/us/en/p/kallax-shelf-unit-white-80275887/",
    "https://www.homedepot.com/p/DEWALT-20V-MAX-Cordless-Drill-Driver-Kit/206525983",
    "https://www.aliexpress.com/item/1005001234567890.html?spm=a2g0o.productlist",
    "https://www.newegg.com/intel-core-i7-14700k-core-i7-14th-gen/p/N82E16819118463",
    "https://www.zappos.com/p/nike-air-max-270-black-anthracite-white/product/8991234",
    "https://www.costco.com/kitchen-appliances.html?refine=||Category_PathHierarchy",

    # 3. Documentation, API References, & Technical Knowledge (20 URLs)
    "https://docs.python.org/3/library/multiprocessing.html#module-multiprocessing",
    "https://docs.python.org/3/howto/logging.html#logging-basic-tutorial",
    "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array/reduce",
    "https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Content-Security-Policy",
    "https://fastapi.tiangolo.com/tutorial/bigger-applications/#an-example-file-structure",
    "https://fastapi.tiangolo.com/tutorial/security/oauth2-jwt/",
    "https://kubernetes.io/docs/concepts/workloads/controllers/deployment/#creating-a-deployment",
    "https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/",
    "https://docs.docker.com/engine/reference/commandline/run/#add-bind-mounts-or-volumes-using-the---mount-flag",
    "https://docs.docker.com/compose/compose-file/05-services/",
    "https://react.dev/reference/react/useEffect#connecting-to-an-external-system",
    "https://react.dev/learn/state-a-components-memory",
    "https://en.wikipedia.org/wiki/Phishing_detection_methods",
    "https://en.wikipedia.org/wiki/Transport_Layer_Security#History_and_development",
    "https://stackoverflow.com/questions/11227809/why-is-processing-a-sorted-array-faster-than-processing-an-unsorted-array",
    "https://stackoverflow.com/questions/2003505/how-do-i-delete-a-git-branch-locally-and-remotely",
    "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html",
    "https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_regression.html",
    "https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBClassifier",
    "https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html",

    # 4. SaaS, Web Apps, Cloud, & AI Platforms (20 URLs)
    "https://claude.ai/chat/a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "https://claude.ai/chat/98765432-fedc-ba09-8765-43210fedcba0",
    "https://chatgpt.com/c/67890abc-def1-2345-6789-0abcdef12345",
    "https://console.cloud.google.com/compute/instances?project=my-cloud-project-123",
    "https://console.cloud.google.com/storage/browser/my-backup-bucket-2026",
    "https://portal.azure.com/#view/HubsExtension/BrowseResource/resourceType/Microsoft.Compute%2FVirtualMachines",
    "https://aws.amazon.com/s3/pricing/?nc1=h_ls&loc=3",
    "https://aws.amazon.com/ec2/instance-types/t4g/",
    "https://app.slack.com/client/T01234567/C09876543/thread/1680000000.123456",
    "https://linear.app/my-team/issue/ENG-1234/fix-phishing-scale-bias-bug",
    "https://notion.so/myworkspace/Engineering-Architecture-Plan-8f7e6d5c4b3a210",
    "https://figma.com/file/aBcDeFgHiJkLmNoP/Design-System-v2?node-id=102%3A405",
    "https://open.spotify.com/playlist/37i9dQZF1DXcBWIGoYBM5M?si=1234567890abcdef",
    "https://music.apple.com/us/album/abbey-road-remastered/401186200",
    "https://trello.com/b/1234abcd/product-roadmap-2026",
    "https://datadoghq.com/product/infrastructure-monitoring/",
    "https://auth0.com/docs/authenticate/login/configure-universal-login-experience",
    "https://stripe.com/docs/payments/accept-a-payment?platform=web&ui=elements",
    "https://cloudflare.com/products/zero-trust/access/",
    "https://mongodb.com/docs/atlas/tutorial/connect-to-your-cluster/",

    # 5. News, Media, Social, Blogs & Research (20 URLs)
    "https://www.nytimes.com/2024/01/15/technology/artificial-intelligence-cybersecurity.html",
    "https://www.nytimes.com/section/world/asia",
    "https://www.bbc.com/news/technology-68123456",
    "https://www.bbc.co.uk/programmes/articles/5W6Q7V8/about-the-show",
    "https://arxiv.org/abs/2304.12345",
    "https://arxiv.org/pdf/2304.12345.pdf",
    "https://medium.com/@author/deep-dive-into-machine-learning-security-c8a7b6d5e4f3",
    "https://medium.com/towards-data-science/mastering-xgboost-hyperparameter-tuning-9876543210ab",
    "https://news.ycombinator.com/item?id=39123456",
    "https://news.ycombinator.com/ask",
    "https://reddit.com/r/MachineLearning/comments/1234567/discussion_real_world_phishing_detection/",
    "https://reddit.com/r/Python/comments/987654/announcing_new_library_version/",
    "https://youtube.com/watch?v=dQw4w9WgXcQ&list=PL1234567890&index=1",
    "https://youtube.com/channel/UC_x5XG1OV2P6uZZ5FSM9Ttw/playlists",
    "https://linkedin.com/in/cybersecurity-expert-123456/recent-activity/all/",
    "https://linkedin.com/company/google/jobs/",
    "https://bloomberg.com/news/articles/2026-08-20/global-tech-market-insights",
    "https://reuters.com/technology/cybersecurity-trends-enterprise-protection-2026-08-25/",
    "https://techcrunch.com/2026/08/15/next-gen-cybersecurity-innovations/",
    "https://theverge.com/tech/2026/8/10/23829104/next-generation-web-browsers-security",
]


def fetch_and_label_urls(
    n_phishing: int = 300,
    n_legitimate: int = 300,
    n_complex_legitimate: int = 100,
    random_state: int = 42,
) -> Tuple[List[str], List[int]]:
    """
    Fetch real phishing and legitimate URLs from the UCI PhiUSIIL Phishing URL
    Dataset (id=967) and augment with realistic complex-path legitimate URLs.

    In the PhiUSIIL dataset:
      - label == 0: Phishing (~101k URLs) -> mapped to target class 1
      - label == 1: Legitimate (~135k URLs) -> mapped to target class 0

    Augmented Legitimate Set:
      - Appends ~100 realistic URLs with complex paths, query strings, UUIDs,
        and deep subdirectories to eliminate path/length distribution bias.

    Parameters
    ----------
    n_phishing : int, default=300
        Number of phishing URLs to sample from PhiUSIIL.
    n_legitimate : int, default=300
        Number of legitimate URLs to sample from PhiUSIIL.
    n_complex_legitimate : int, default=100
        Number of complex-path legitimate URLs to append.
    random_state : int, default=42
        Random seed for reproducible sampling.

    Returns
    -------
    tuple[list[str], list[int]]
        (urls, labels) where 1 indicates phishing and 0 indicates legitimate.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    df_urls: Optional[pd.DataFrame] = None

    # Check local cache first for fast repeat runs
    if CACHE_FILE.exists():
        try:
            logger.info(f"Loading cached PhiUSIIL raw URLs from {CACHE_FILE}...")
            df_urls = pd.read_parquet(CACHE_FILE)
            logger.info(f"Loaded {len(df_urls)} URLs from cache.")
        except Exception as exc:
            logger.warning(f"Failed to read cache file: {exc}. Re-fetching from UCI repository...")
            df_urls = None

    if df_urls is None or df_urls.empty:
        logger.info("Fetching PhiUSIIL Phishing URL Dataset (id=967) via ucimlrepo...")
        from ucimlrepo import fetch_ucirepo

        dataset = fetch_ucirepo(id=967)
        if hasattr(dataset.data, "original") and dataset.data.original is not None:
            raw_df = dataset.data.original[["URL", "label"]].copy()
        else:
            raw_df = pd.concat([dataset.data.features["URL"], dataset.data.targets["label"]], axis=1)

        # Clean string URLs
        raw_df["URL"] = raw_df["URL"].astype(str).str.strip()
        raw_df = raw_df.dropna(subset=["URL", "label"])
        raw_df = raw_df[raw_df["URL"].str.len() > 0]
        df_urls = raw_df

        # Save to local cache
        try:
            df_urls.to_parquet(CACHE_FILE, index=False)
            logger.info(f"Cached {len(df_urls)} PhiUSIIL raw URLs to {CACHE_FILE}.")
        except Exception as exc:
            logger.warning(f"Could not cache to parquet: {exc}")

    # PhiUSIIL: label == 0 is Phishing, label == 1 is Legitimate
    phishing_subset = df_urls[df_urls["label"] == 0]["URL"]
    legitimate_subset = df_urls[df_urls["label"] == 1]["URL"]

    # Sample required counts
    phishing_sample = phishing_subset.sample(n=min(n_phishing, len(phishing_subset)), random_state=random_state).tolist()
    legitimate_sample = legitimate_subset.sample(n=min(n_legitimate, len(legitimate_subset)), random_state=random_state).tolist()

    # Add complex-path legitimate URLs
    complex_sample = COMPLEX_LEGITIMATE_URLS[:n_complex_legitimate]

    logger.info(
        f"Dataset composition: {len(phishing_sample)} Phishing URLs (label=1), "
        f"{len(legitimate_sample)} Standard Legitimate URLs (label=0), "
        f"{len(complex_sample)} Complex-Path Legitimate URLs (label=0)."
    )

    combined_urls = phishing_sample + legitimate_sample + complex_sample
    combined_labels = [1] * len(phishing_sample) + [0] * (len(legitimate_sample) + len(complex_sample))

    return combined_urls, combined_labels


def build_real_training_dataset(
    urls: List[str],
    labels: List[int],
    visual_reference_dir: Optional[str] = None,
    use_cache: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extract real-world structural features for a list of URLs and fuse them into
    a machine learning feature matrix. Caches extracted feature vectors to
    parquet to accelerate subsequent model initializations.

    Parameters
    ----------
    urls : list[str]
        List of URL strings to process.
    labels : list[int]
        Corresponding binary class labels (1=phishing, 0=legitimate).
    visual_reference_dir : Optional[str]
        Optional path to brand reference logo library.
    use_cache : bool, default=True
        Whether to load from/save to persistent parquet feature cache.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        (X, y) where X is the 16-feature DataFrame and y is the binary labels Series.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Check if feature cache exists and contains all required URLs
    if use_cache and FEATURE_CACHE_FILE.exists():
        try:
            cached_df = pd.read_parquet(FEATURE_CACHE_FILE)
            if len(cached_df) == len(urls) and "label" in cached_df.columns:
                logger.info(f"Loaded {len(cached_df)} extracted feature vectors from cache: {FEATURE_CACHE_FILE}")
                X = cached_df.drop(columns=["label", "url"], errors="ignore")
                y = cached_df["label"].astype(int)
                return X, y
        except Exception as exc:
            logger.warning(f"Could not load feature cache: {exc}. Performing live feature extraction...")

    records: List[dict] = []
    valid_labels: List[int] = []
    valid_urls: List[str] = []
    total = len(urls)

    # Neutral visual and content feature baselines for URL-only benchmark dataset
    neutral_visual_features = {
        "closest_match": None,
        "hash_distance": 64.0,
        "is_visually_similar": False,
        "all_distances": {},
    }
    neutral_content_features = {
        "urgency_score": 0.0,
        "credential_request_score": 0.0,
        "login_prompt_density": 0.0,
        "tfidf_top_terms": [],
        "overall_content_risk_score": 0.0,
    }

    t0 = time.time()
    logger.info(f"Starting real feature extraction across {total} URLs...")

    for idx, (url, label) in enumerate(zip(urls, labels), start=1):
        try:
            # 1. Live structural feature extraction (includes WHOIS lookup, entropy, etc.)
            url_feats = extract_url_features(url)

            # 2. Fuse into unified 16-dimensional feature vector
            vector = build_feature_vector(
                url_features=url_feats,
                visual_features=neutral_visual_features,
                content_features=neutral_content_features,
            )

            records.append(vector)
            valid_labels.append(label)
            valid_urls.append(url)
        except Exception as exc:
            logger.warning(f"Error extracting features for URL '{url}': {exc}")
            continue

        if idx % 50 == 0 or idx == total:
            elapsed = time.time() - t0
            avg_per_url = elapsed / idx if idx > 0 else 0
            print(
                f"  [Feature Extraction] Processed {idx}/{total} URLs "
                f"({(idx/total)*100:.1f}%) | Elapsed: {elapsed:.1f}s ({avg_per_url:.2f}s/URL)"
            )

    X = pd.DataFrame(records)
    y = pd.Series(valid_labels, name="label")

    # Cache extracted features to parquet
    if use_cache:
        try:
            cache_save_df = X.copy()
            cache_save_df["label"] = valid_labels
            cache_save_df["url"] = valid_urls
            cache_save_df.to_parquet(FEATURE_CACHE_FILE, index=False)
            logger.info(f"Cached {len(cache_save_df)} extracted feature vectors to {FEATURE_CACHE_FILE}.")
        except Exception as exc:
            logger.warning(f"Failed to write feature cache: {exc}")

    logger.info(f"Feature extraction completed for {len(X)} URLs in {time.time() - t0:.2f}s.")
    return X, y
