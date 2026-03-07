"""
Flickr8k Dataset Loader for Image Captioning
=============================================
Expects the following directory structure:

    flickr8k/
    ├── Images/        # all .jpg images
    └── captions.txt   # CSV: image,caption  (with header row)

Example captions.txt rows:
    image,caption
    1000268201_693b08cb0e.jpg,A child in a pink dress ...
    1000268201_693b08cb0e.jpg,A girl going into a wooden building .

No split .txt files needed — the loader auto-generates a reproducible
70 / 15 / 15  train / val / test split by image, so all 5 captions
for a given image always land in the same split.

Install dependencies:
    pip install torch torchvision pillow pandas
"""

import os
import re
import torch
import pandas as pd
from PIL import Image
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ──────────────────────────────────────────────
# 1.  VOCABULARY
# ──────────────────────────────────────────────

class Vocabulary:
    """
    Word-level vocabulary built from training captions.

    Special tokens
    --------------
    <PAD>  index 0  - padding
    <SOS>  index 1  - start of sequence
    <EOS>  index 2  - end of sequence
    <UNK>  index 3  - unknown / rare word
    """

    PAD_TOKEN = "<PAD>"
    SOS_TOKEN = "<SOS>"
    EOS_TOKEN = "<EOS>"
    UNK_TOKEN = "<UNK>"

    def __init__(self, freq_threshold: int = 5):
        self.freq_threshold = freq_threshold
        self.stoi = {
            self.PAD_TOKEN: 0,
            self.SOS_TOKEN: 1,
            self.EOS_TOKEN: 2,
            self.UNK_TOKEN: 3,
        }
        self.itos = {v: k for k, v in self.stoi.items()}

    def __len__(self) -> int:
        return len(self.stoi)

    @staticmethod
    def tokenize(text: str) -> list:
        """Lowercase and split on non-alphanumeric characters."""
        return re.findall(r"\w+", text.lower())

    def build(self, captions: list) -> None:
        """Populate vocabulary from a list of caption strings."""
        counter = Counter()
        for caption in captions:
            counter.update(self.tokenize(caption))

        idx = len(self.stoi)
        for word, freq in counter.items():
            if freq >= self.freq_threshold:
                self.stoi[word] = idx
                self.itos[idx]  = word
                idx += 1

        print(f"[Vocabulary] {len(self)} tokens  "
              f"(freq_threshold={self.freq_threshold})")

    def encode(self, caption: str) -> list:
        """Caption string -> list of token indices."""
        unk = self.stoi[self.UNK_TOKEN]
        return [self.stoi.get(t, unk) for t in self.tokenize(caption)]

    def decode(self, indices: list) -> str:
        """Token indices -> caption string (special tokens stripped)."""
        skip = {self.SOS_TOKEN, self.EOS_TOKEN, self.PAD_TOKEN}
        return " ".join(
            self.itos.get(i, self.UNK_TOKEN)
            for i in indices
            if self.itos.get(i, self.UNK_TOKEN) not in skip
        )


# ──────────────────────────────────────────────
# 2.  CAPTION FILE PARSER
# ──────────────────────────────────────────────

def _load_captions_df(captions_path: str) -> pd.DataFrame:
    """
    Parse a CSV captions file in the format:

        image,caption
        1000268201_693b08cb0e.jpg,A child in a pink dress ...
        1000268201_693b08cb0e.jpg,A girl going into a wooden ...

    Returns a DataFrame with columns ['image', 'caption'].
    """
    df = pd.read_csv(captions_path)

    # normalise column names in case of leading/trailing whitespace
    df.columns = [c.strip().lower() for c in df.columns]

    if "image" not in df.columns or "caption" not in df.columns:
        raise ValueError(
            f"Expected columns 'image' and 'caption', "
            f"but found: {list(df.columns)}\n"
            f"Check that captions.txt is comma-separated with a header row."
        )

    df = (
        df[["image", "caption"]]
        .dropna()
        .reset_index(drop=True)
    )
    print(f"[Parser] Loaded {len(df)} caption rows "
          f"for {df['image'].nunique()} unique images.")
    return df


# ──────────────────────────────────────────────
# 3.  SPLIT GENERATOR
# ──────────────────────────────────────────────

def _make_splits(
    df:         pd.DataFrame,
    train_frac: float = 0.70,
    val_frac:   float = 0.15,
    seed:       int   = 42,
) -> dict:
    """
    Randomly assign unique images to train / val / test splits.

    Splitting by image ensures all 5 captions for one image always
    land in the same split — important for correct BLEU evaluation.
    The split is deterministic for a given seed.

    With Flickr8k's 8,000 images and 70/15/15 fractions:
        train  ~5,600 images  -> ~28,000 caption pairs
        val    ~1,200 images  ->  ~6,000 caption pairs
        test   ~1,200 images  ->  ~6,000 caption pairs

    Returns a dict with keys 'train', 'val', 'test' mapping to
    sets of image filenames.
    """
    images = sorted(df["image"].unique())       # sort first for reproducibility
    rng    = torch.Generator().manual_seed(seed)
    perm   = torch.randperm(len(images), generator=rng).tolist()

    n       = len(images)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)

    train_imgs = {images[i] for i in perm[:n_train]}
    val_imgs   = {images[i] for i in perm[n_train : n_train + n_val]}
    test_imgs  = {images[i] for i in perm[n_train + n_val :]}

    print(f"[Splits] train={len(train_imgs)}  "
          f"val={len(val_imgs)}  test={len(test_imgs)}  images  "
          f"(70/15/15,  seed={seed})")
    return {"train": train_imgs, "val": val_imgs, "test": test_imgs}


# ──────────────────────────────────────────────
# 4.  DATASET
# ──────────────────────────────────────────────

class Flickr8kDataset(Dataset):
    """
    PyTorch Dataset for Flickr8k image captioning.

    Each __getitem__ returns one (image_tensor, caption_tensor) pair.
    Because every image has 5 reference captions, the dataset length is
    5 x number_of_images_in_split.

    Parameters
    ----------
    root_dir       : str              path to the flickr8k/ folder
    split          : str              'train', 'val', or 'test'
    vocab          : Vocabulary|None  pass None only for 'train';
                                      val/test must reuse train vocab
    transform      : callable|None    torchvision transform for images
    freq_threshold : int              min word frequency for vocabulary
    train_frac     : float            fraction of images -> train
    val_frac       : float            fraction of images -> val
    seed           : int              RNG seed for reproducible splits
    """

    def __init__(
        self,
        root_dir,
        split          = "train",
        vocab          = None,
        transform      = None,
        freq_threshold = 5,
        train_frac     = 0.70,
        val_frac       = 0.15,
        seed           = 42,
    ):
        assert split in ("train", "val", "test"), \
            "split must be 'train', 'val', or 'test'"

        self.root_dir  = root_dir
        self.img_dir   = os.path.join(root_dir, "Images")
        self.split     = split
        self.transform = transform or self._default_transform()

        # ── Load & parse captions ──────────────────────────────────────
        captions_path = os.path.join(root_dir, "captions.txt")
        df = _load_captions_df(captions_path)

        # ── Apply split ────────────────────────────────────────────────
        splits     = _make_splits(df, train_frac, val_frac, seed)
        split_imgs = splits[split]
        df         = df[df["image"].isin(split_imgs)].reset_index(drop=True)

        self.images   = df["image"].tolist()
        self.captions = df["caption"].tolist()

        print(f"[Flickr8kDataset] split='{split}'  "
              f"caption pairs={len(self)}")

        # ── Vocabulary ─────────────────────────────────────────────────
        if vocab is None:
            # Build from training captions only — never from val/test
            self.vocab = Vocabulary(freq_threshold)
            self.vocab.build(self.captions)
        else:
            self.vocab = vocab

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        """
        Returns
        -------
        image   : FloatTensor  (3, 224, 224)
        caption : LongTensor   (T,)   includes <SOS> and <EOS>
        """
        # ── Image ──────────────────────────────────────────────────────
        img_path = os.path.join(self.img_dir, self.images[idx])
        image    = Image.open(img_path).convert("RGB")
        image    = self.transform(image)

        # ── Caption ────────────────────────────────────────────────────
        indices = (
            [self.vocab.stoi[Vocabulary.SOS_TOKEN]]
            + self.vocab.encode(self.captions[idx])
            + [self.vocab.stoi[Vocabulary.EOS_TOKEN]]
        )
        caption = torch.tensor(indices, dtype=torch.long)

        return image, caption

    # ------------------------------------------------------------------
    @staticmethod
    def _default_transform():
        """
        Standard ImageNet preprocessing.
        Compatible with ResNet, EfficientNet, and VGG encoders.
        """
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225],
            ),
        ])


# ──────────────────────────────────────────────
# 5.  COLLATE  (pads variable-length captions)
# ──────────────────────────────────────────────

class CaptionCollate:
    """
    Pads captions in a batch to the length of the longest one.

    Returns
    -------
    images   : FloatTensor  (B, 3, 224, 224)
    captions : LongTensor   (B, max_seq_len)  — zero-padded with <PAD>
    lengths  : list[int]    original unpadded lengths; needed for
                            pack_padded_sequence in the LSTM decoder
    """

    def __init__(self, pad_idx=0):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        images, captions = zip(*batch)
        images  = torch.stack(images, dim=0)
        lengths = [cap.size(0) for cap in captions]
        max_len = max(lengths)

        padded = torch.full(
            (len(captions), max_len),
            fill_value=self.pad_idx,
            dtype=torch.long,
        )
        for i, cap in enumerate(captions):
            padded[i, : cap.size(0)] = cap

        return images, padded, lengths


# ──────────────────────────────────────────────
# 6.  CONVENIENCE FACTORY
# ──────────────────────────────────────────────

def get_flickr8k_loaders(
    root_dir,
    batch_size     = 32,
    num_workers    = 4,
    freq_threshold = 5,
    train_frac     = 0.70,
    val_frac       = 0.15,
    seed           = 42,
    transform      = None,
):
    """
    Build train / val / test DataLoaders in one call.

    Vocabulary is built from training captions only, then shared
    across all splits so token indices are consistent.

    Parameters
    ----------
    root_dir       : path to flickr8k/ folder
    batch_size     : images per batch
    num_workers    : DataLoader worker processes (use 4+ on Discovery)
    freq_threshold : min word frequency for vocabulary inclusion
    train_frac     : fraction of images for training  (default 0.70)
    val_frac       : fraction of images for validation (default 0.15)
                     remainder (0.15) -> test
    seed           : RNG seed for reproducible splits
    transform      : optional custom torchvision transform

    Returns
    -------
    train_loader, val_loader, test_loader, vocab
    """
    shared_kwargs = dict(
        root_dir   = root_dir,
        transform  = transform,
        train_frac = train_frac,
        val_frac   = val_frac,
        seed       = seed,
    )

    # Vocab built from training data only
    train_ds = Flickr8kDataset(
        split="train", vocab=None,
        freq_threshold=freq_threshold,
        **shared_kwargs,
    )
    vocab = train_ds.vocab

    val_ds  = Flickr8kDataset(split="val",  vocab=vocab, **shared_kwargs)
    test_ds = Flickr8kDataset(split="test", vocab=vocab, **shared_kwargs)

    collate_fn = CaptionCollate(pad_idx=vocab.stoi[Vocabulary.PAD_TOKEN])

    def _loader(ds, shuffle):
        return DataLoader(
            ds,
            batch_size  = batch_size,
            shuffle     = shuffle,
            num_workers = num_workers,
            collate_fn  = collate_fn,
            pin_memory  = torch.cuda.is_available(),
        )

    return (
        _loader(train_ds, shuffle=True),
        _loader(val_ds,   shuffle=False),
        _loader(test_ds,  shuffle=False),
        vocab,
    )


# ──────────────────────────────────────────────
# 7.  SMOKE TEST
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    ROOT = sys.argv[1] if len(sys.argv) > 1 else "./flickr8k"

    train_loader, val_loader, test_loader, vocab = get_flickr8k_loaders(
        root_dir       = ROOT,
        batch_size     = 32,
        num_workers    = 0,     # 0 for quick local test; use 4 on Discovery
        freq_threshold = 5,
    )

    print(f"\nVocabulary size : {len(vocab)}")
    print(f"Train batches   : {len(train_loader)}")
    print(f"Val   batches   : {len(val_loader)}")
    print(f"Test  batches   : {len(test_loader)}")

    images, captions, lengths = next(iter(train_loader))
    print(f"\nBatch shapes")
    print(f"  images   : {tuple(images.shape)}")    # (32, 3, 224, 224)
    print(f"  captions : {tuple(captions.shape)}")  # (32, max_seq_len)
    print(f"  lengths  : {lengths[:5]}  ...")

    first_cap = captions[0][: lengths[0]].tolist()
    print(f"\nDecoded sample caption:\n  {vocab.decode(first_cap)}")