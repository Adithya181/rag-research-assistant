"""
Download the 8 ML papers used by the RAG pipeline into data/papers/.
Run this once before starting the app:

    python scripts/download_papers.py
"""
import os
import urllib.request

PAPERS = {
    "attention_is_all_you_need.pdf": "https://arxiv.org/pdf/1706.03762",
    "bert.pdf": "https://arxiv.org/pdf/1810.04805",
    "resnet.pdf": "https://arxiv.org/pdf/1512.03385",
    "gpt3.pdf": "https://arxiv.org/pdf/2005.14165",
    "adam_optimizer.pdf": "https://arxiv.org/pdf/1412.6980",
    "dropout.pdf": "https://arxiv.org/pdf/1207.0580",
    "batch_normalization.pdf": "https://arxiv.org/pdf/1502.03167",
    "vit.pdf": "https://arxiv.org/pdf/2010.11929",
}


def download_papers(target_dir: str = "data/papers") -> None:
    os.makedirs(target_dir, exist_ok=True)
    for filename, url in PAPERS.items():
        path = os.path.join(target_dir, filename)
        if os.path.exists(path):
            print(f"Already exists: {filename}")
            continue
        urllib.request.urlretrieve(url, path)
        print(f"Downloaded: {filename}")
    print(f"\nTotal papers in {target_dir}: {len(os.listdir(target_dir))}")


if __name__ == "__main__":
    download_papers()
