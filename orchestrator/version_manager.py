"""
Version Manager — search-before-save versioning.
Checks if similar content already exists via CHUNKS_LEXICAL search,
then appends version metadata tags to the text before cognify.
"""
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Matches [VERSION:N] tags we append to text
_VERSION_TAG_RE = re.compile(r"\[VERSION:(\d+)\]")


@dataclass
class VersionResult:
    action: str           # "create" | "update"
    version_number: int   # 1 for new documents, 2+ for updates
    dataset_name: str
    versioned_text: str   # text with version metadata appended


async def check_and_version(text: str, dataset_name: str) -> VersionResult:
    """
    1. Fast lexical search for existing similar content.
    2. If found, extract the highest version number and increment.
    3. Append version metadata tags to text.
    """
    from orchestrator.mcp_client import wrapper

    # Use the first 200 chars as a representative fingerprint
    fingerprint = text[:200].strip()

    existing_version = 0
    try:
        result = await wrapper.call(
            "search",
            {
                "search_query": fingerprint,
                "search_type": "CHUNKS_LEXICAL",
            },
            timeout=30,
        )
        if result:
            # Scan all [VERSION:N] tags in the search results
            matches = _VERSION_TAG_RE.findall(result)
            if matches:
                existing_version = max(int(v) for v in matches)
                logger.debug("Found existing version %d for dataset '%s'", existing_version, dataset_name)
    except Exception as exc:
        logger.warning("Version check search failed (%s) — treating as new document", exc)

    version_number = existing_version + 1
    action = "update" if existing_version > 0 else "create"

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    metadata_tag = (
        f"\n\n[VERSION:{version_number}]"
        f"[DATASET:{dataset_name}]"
        f"[UPDATED:{timestamp}]"
    )
    versioned_text = text + metadata_tag

    logger.info(
        "Version %s: document '%s...' → v%d in %s",
        action, fingerprint[:60], version_number, dataset_name,
    )

    return VersionResult(
        action=action,
        version_number=version_number,
        dataset_name=dataset_name,
        versioned_text=versioned_text,
    )
