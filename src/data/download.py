"""Download Amazon Reviews 2023 All_Beauty dataset from HuggingFace mirror."""

import os
import urllib.request

HF_MIRROR = "https://hf-mirror.com/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main"

FILES = {
    "reviews": "raw/review_categories/All_Beauty.jsonl",
    "metadata": "raw_meta_All_Beauty/full-00000-of-00001.parquet",
}


def download_file(url: str, save_path: str) -> None:
    """Download a file with progress display."""
    if os.path.exists(save_path):
        size = os.path.getsize(save_path)
        print(f"Already exists: {save_path} ({size:,} bytes)")
        return
    print(f"Downloading: {url}")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    tmp_path = save_path + ".tmp"

    def report(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(downloaded * 100 // total_size, 100)
            print(f"\r  Progress: {pct}% ({downloaded:,}/{total_size:,})", end="", flush=True)

    urllib.request.urlretrieve(url, tmp_path, reporthook=report)
    print()
    os.rename(tmp_path, save_path)
    size = os.path.getsize(save_path)
    print(f"Saved: {save_path} ({size:,} bytes)")


def download_all(data_dir: str = "./AmazonReviews2023") -> dict[str, str]:
    """Download all required data files. Returns paths dict."""
    paths = {}
    for key, remote_path in FILES.items():
        url = f"{HF_MIRROR}/{remote_path}"
        filename = os.path.basename(remote_path)
        save_path = os.path.join(data_dir, filename)
        download_file(url, save_path)
        paths[key] = save_path
    return paths


if __name__ == "__main__":
    paths = download_all()
    for k, v in paths.items():
        print(f"{k}: {v}")
