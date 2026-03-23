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

Split sizes (Flickr8k specific):
    train : 6000 images -> 30,000 caption pairs
    val   : 1000 images ->  5,000 caption pairs
    test  : remaining   -> ~5,000 caption pairs

Caption length:
    All captions are padded to the same fixed length (max_caption_length)
    BEFORE adding <SOS> and <EOS> tokens. This means every caption tensor
    has the same size: max_caption_length + 2.

Vocabulary:
    Every unique word gets a token — no frequency threshold.
    Built ONLY from training captions to prevent data leakage.
    The same vocabulary object is then shared with val and test splits.

Default image transform:
    Uses ImageNet mean and std because our encoder (ResNet/EfficientNet)
    was pretrained on ImageNet — this puts images in the distribution
    the encoder already understands.

Install dependencies:
    pip install torch torchvision pillow pandas

Usage:
    python Dataloader.py --root ./flickr8k
    python Dataloader.py --root ./flickr8k --image_path ./flickr8k/Images/some_image.jpg
"""

import os
import re
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ──────────────────────────────────────────────
# 1.  VOCABULARY
# ──────────────────────────────────────────────

class Vocabulary:
    """
    Word-level vocabulary built from training captions.

    Maintains two lookup tables:
        word_to_index : maps a word string -> its integer token ID
        index_to_word : maps an integer ID -> its word string

    Special tokens (always present, assigned first):
        <PAD>  index 0  — padding token, makes all captions the same length
        <SOS>  index 1  — start-of-sequence token, prepended to every caption
        <EOS>  index 2  — end-of-sequence token, appended to every caption
        <UNK>  index 3  — unknown token, for words not seen during training
    """

    PAD_TOKEN = "<PAD>"
    SOS_TOKEN = "<SOS>"
    EOS_TOKEN = "<EOS>"
    UNK_TOKEN = "<UNK>"

    def __init__(self):
        # word_to_index: {"dog": 4, "running": 5, "<PAD>": 0, ...}
        self.word_to_index = {
            self.PAD_TOKEN: 0,
            self.SOS_TOKEN: 1,
            self.EOS_TOKEN: 2,
            self.UNK_TOKEN: 3,
        }
        # index_to_word: reverse lookup {0: "<PAD>", 4: "dog", ...}
        self.index_to_word = {index: word for word, index in self.word_to_index.items()}

    def __len__(self) -> int:
        return len(self.word_to_index)

    @staticmethod
    def tokenize(text: str) -> list:
        """
        Split a caption string into a list of lowercase word tokens.
        e.g. "A Dog Runs!" -> ["a", "dog", "runs"]
        Uses regex to keep only alphanumeric characters (strips punctuation).
        """
        return re.findall(r"\w+", text.lower())

    def build(self, captions: list) -> None:
        """
        Build the vocabulary from a list of caption strings.
        Every unique word gets a token — no frequency filtering.
        Only called once, using TRAINING captions only.
        """
        # Collect all unique words across all captions
        unique_words = set()
        for caption in captions:
            unique_words.update(self.tokenize(caption))

        # Assign an index to each unique word, starting after the 4 special tokens
        next_index = len(self.word_to_index)
        for word in sorted(unique_words):   # sorted for reproducibility
            self.word_to_index[word] = next_index
            self.index_to_word[next_index] = word
            next_index += 1

        print(f"[Vocabulary] {len(self)} tokens total")

    def encode(self, caption: str) -> list:
        """
        Convert a caption string into a list of integer token IDs.
        e.g. "a dog runs" -> [4, 5, 6]

        Words not in vocabulary map to <UNK> (index 3).
        <SOS> and <EOS> are NOT added here — added separately in __getitem__.
        """
        unknown_index = self.word_to_index[self.UNK_TOKEN]
        return [
            self.word_to_index.get(word, unknown_index)
            for word in self.tokenize(caption)
        ]

    def decode(self, indices: list) -> str:
        """
        Convert a list of integer token IDs back into a readable caption string.
        e.g. [1, 4, 5, 6, 2, 0, 0] -> "a dog runs"

        Special tokens (<SOS>, <EOS>, <PAD>) are stripped from the output.
        """
        special_tokens = {self.SOS_TOKEN, self.EOS_TOKEN, self.PAD_TOKEN}
        words = []
        for index in indices:
            word = self.index_to_word.get(index, self.UNK_TOKEN)
            if word not in special_tokens:
                words.append(word)
        return " ".join(words)


# ──────────────────────────────────────────────
# 2.  CAPTION FILE PARSER
# ──────────────────────────────────────────────

def _load_captions_df(captions_path: str) -> pd.DataFrame:
    """
    Parse captions.txt into a DataFrame with columns ['image', 'caption'].
    Each image has 5 rows (one per reference caption).
    """
    df = pd.read_csv(captions_path)

    # Normalise column names in case of leading/trailing whitespace
    df.columns = [col.strip().lower() for col in df.columns]

    if "image" not in df.columns or "caption" not in df.columns:
        raise ValueError(
            f"Expected columns 'image' and 'caption', "
            f"but found: {list(df.columns)}\n"
            f"Check that captions.txt is comma-separated with a header row."
        )

    df = df[["image", "caption"]].dropna().reset_index(drop=True)

    print(f"[Parser] Loaded {len(df)} caption rows "
          f"for {df['image'].nunique()} unique images.")
    return df


# ──────────────────────────────────────────────
# 3.  SPLIT GENERATOR
# ──────────────────────────────────────────────

def _make_splits(
    df:      pd.DataFrame,
    n_train: int = 6000,
    n_val:   int = 1000,
    seed:    int = 42,
) -> dict:
    """
    Assign unique images to train / val / test splits.

    Flickr8k specific sizes:
        train : 6000 images -> 30,000 caption pairs
        val   : 1000 images ->  5,000 caption pairs
        test  : remaining   -> ~5,000 caption pairs

    WHY split by image (not by row)?
        Each image has 5 caption rows. Splitting by row could put the same
        image in both train and test, making evaluation meaningless.
        Splitting by image guarantees all 5 captions for one image always
        land in the same split.

    Seed ensures the same split every run (reproducibility).
    """
    all_images = sorted(df["image"].unique())
    total = len(all_images)

    if n_train + n_val >= total:
        raise ValueError(
            f"n_train ({n_train}) + n_val ({n_val}) must be less than "
            f"total images ({total})."
        )

    # Reproducible shuffle using fixed seed
    rng = torch.Generator().manual_seed(seed)
    shuffled_indices = torch.randperm(total, generator=rng).tolist()

    train_images = {all_images[i] for i in shuffled_indices[:n_train]}
    val_images   = {all_images[i] for i in shuffled_indices[n_train : n_train + n_val]}
    test_images  = {all_images[i] for i in shuffled_indices[n_train + n_val :]}

    print(f"[Splits] train={len(train_images)}  "
          f"val={len(val_images)}  test={len(test_images)}  images  "
          f"(seed={seed})")
    return {"train": train_images, "val": val_images, "test": test_images}


# ──────────────────────────────────────────────
# 4.  DATASET
# ──────────────────────────────────────────────

class Flickr8kDataset(Dataset):
    """
    PyTorch Dataset for Flickr8k image captioning.

    Each __getitem__ returns one (image_tensor, caption_tensor) pair.

    Caption length:
        All captions are padded to max_caption_length using <PAD> tokens,
        THEN <SOS> is prepended and <EOS> is appended.
        Final caption tensor length = max_caption_length + 2 (same for all).

    How captions are selected:
        self.images and self.captions are flat parallel lists —
        every row is one (image, caption) pair. __getitem__(idx) returns
        a specific deterministic pair. Over a full epoch the model sees
        all 5 captions for every image.

    Parameters
    ----------
    root_dir           : str              path to the flickr8k/ folder
    split              : str              'train', 'val', or 'test'
    max_caption_length : int              all captions padded to this length
                                          before adding <SOS> and <EOS>
    vocab              : Vocabulary|None  pass None only for 'train'
    transform          : callable|None    torchvision transform for images
    n_train            : int              number of images for train split
    n_val              : int              number of images for val split
    seed               : int              RNG seed for reproducible splits
    """

    def __init__(
        self,
        root_dir,
        split              = "train",
        max_caption_length = 30,
        vocab              = None,
        transform          = None,
        n_train            = 6000,
        n_val              = 1000,
        seed               = 42,
    ):
        assert split in ("train", "val", "test"), \
            "split must be 'train', 'val', or 'test'"

        self.root_dir           = root_dir
        self.img_dir            = os.path.join(root_dir, "Images")
        self.split              = split
        self.max_caption_length = max_caption_length
        self.transform          = transform or self._default_transform()

        # ── Load & parse captions ──────────────────────────────────────
        captions_path = os.path.join(root_dir, "captions.txt")
        df = _load_captions_df(captions_path)

        # ── Apply split ────────────────────────────────────────────────
        splits       = _make_splits(df, n_train, n_val, seed)
        split_images = splits[split]
        df           = df[df["image"].isin(split_images)].reset_index(drop=True)

        # Flat parallel lists: self.images[i] and self.captions[i] are paired
        self.images   = df["image"].tolist()
        self.captions = df["caption"].tolist()

        print(f"[Flickr8kDataset] split='{split}'  "
              f"caption pairs={len(self)}")

        # ── Vocabulary ─────────────────────────────────────────────────
        if vocab is None:
            # Build vocabulary from training captions ONLY
            if split != "train":
                raise ValueError(
                    "vocab=None is only allowed for split='train'. "
                    "Pass the training vocab to val/test datasets."
                )
            self.vocab = Vocabulary()
            self.vocab.build(self.captions)
        else:
            # Val and test reuse the vocabulary built from training data
            self.vocab = vocab

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        """
        Returns one (image_tensor, caption_tensor) pair at position idx.

        Caption format:
            1. Encode caption words to token IDs
            2. Truncate to max_caption_length if too long
            3. Prepend <SOS> and append <EOS>
            4. Pad with <PAD> to fixed length max_caption_length + 2
            Final length = max_caption_length + 2  (same for every item)
            Final order : <SOS> word1 ... wordN <EOS> <PAD> ... <PAD>

        Returns
        -------
        image   : FloatTensor  (3, 224, 224)
        caption : LongTensor   (max_caption_length + 2,)
        """
        # ── Image ──────────────────────────────────────────────────────
        img_path = os.path.join(self.img_dir, self.images[idx])
        image    = Image.open(img_path).convert("RGB")
        image    = self.transform(image)   # -> FloatTensor (3, 224, 224)

        # ── Caption ────────────────────────────────────────────────────
        # Step 1: encode words to token IDs
        token_ids = self.vocab.encode(self.captions[idx])

        # Step 2: truncate if too long
        pad_index = self.vocab.word_to_index[Vocabulary.PAD_TOKEN]
        token_ids = token_ids[:self.max_caption_length]

        # Step 3: wrap with <SOS> at start and <EOS> immediately after last word
        token_ids = (
            [self.vocab.word_to_index[Vocabulary.SOS_TOKEN]]
            + token_ids
            + [self.vocab.word_to_index[Vocabulary.EOS_TOKEN]]
        )

        # Step 4: pad to fixed length (max_caption_length + 2)
        pad_len = self.max_caption_length + 2 - len(token_ids)
        token_ids = token_ids + [pad_index] * pad_len

        caption = torch.tensor(token_ids, dtype=torch.long)

        # we need to also return image name so in evaluation we know which captions to go to image
        return image, caption, self.images[idx]
    
    def get_all_references(self, split="test"):
        """ 
        When we comput BLEU scores we need to know ALL captions for each image. Without
        this functionm, that is not possible because image shows up multiple times in self.images
        (each with a different caption)

        Here we return a list of list of list of strings which is format needed by our eval metric to get BLEU scores
        """
        references = []

        ref_dict = self.get_all_references_dict()

        for _, captions in ref_dict.items():
            references.append(captions)
         
        return references
    
    def get_all_references_dict(self):
        """ 
        When we comput BLEU scores we need to know ALL captions for each image. Without
        this functionm, that is not possible because image shows up multiple times in self.images
        (each with a different caption)

        Here we return a dictionary of image name to list of captions
        """
        references = {}

        for i, image_name in enumerate(self.images):
            if image_name in references:
                references[image_name].append(self.vocab.tokenize(self.captions[i]))
            else:
                references[image_name] = [self.vocab.tokenize(self.captions[i])]
         
        return references
    

    # ------------------------------------------------------------------
    @staticmethod
    def _default_transform():
        """
        Standard preprocessing for images going into a pretrained CNN encoder.

        Resize: ResNet and EfficientNet both expect 224x224 input.

        Normalize mean/std: Values from ImageNet (NOT Flickr8k specific).
        We use ImageNet stats because our encoder was pretrained on ImageNet —
        normalising our images the same way puts them in the distribution
        the encoder already understands.
        """
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],   # ImageNet channel means (R, G, B)
                std =[0.229, 0.224, 0.225],   # ImageNet channel stds  (R, G, B)
            ),
        ])


# ──────────────────────────────────────────────
# 5.  CONVENIENCE FACTORY
# ──────────────────────────────────────────────

def get_flickr8k_loaders(
    root_dir,
    batch_size         = 32,
    num_workers        = 0,
    max_caption_length = 30,
    n_train            = 6000,
    n_val              = 1000,
    seed               = 42,
    transform          = None,
):
    """
    Build train / val / test DataLoaders in one call.

    Vocabulary is built from training captions only, then shared
    across all splits so token indices are consistent.

    Returns
    -------
    train_loader, val_loader, test_loader, vocab
    """
    shared_kwargs = dict(
        root_dir           = root_dir,
        max_caption_length = max_caption_length,
        transform          = transform,
        n_train            = n_train,
        n_val              = n_val,
        seed               = seed,
    )

    # Build vocab from training data only — pass it to val/test
    train_dataset = Flickr8kDataset(split="train", vocab=None, **shared_kwargs)
    vocab         = train_dataset.vocab

    val_dataset  = Flickr8kDataset(split="val",  vocab=vocab, **shared_kwargs)
    test_dataset = Flickr8kDataset(split="test", vocab=vocab, **shared_kwargs)

    def _make_loader(dataset, shuffle):
        return DataLoader(
            dataset,
            batch_size  = batch_size,
            shuffle     = shuffle,
            num_workers = num_workers,
            pin_memory  = torch.cuda.is_available(),
        )

    return (
        _make_loader(train_dataset, shuffle=True),
        _make_loader(val_dataset,   shuffle=False),
        _make_loader(test_dataset,  shuffle=False),
        vocab,
    )


# ──────────────────────────────────────────────
# 6.  SMOKE TEST
# ──────────────────────────────────────────────
# Run directly to verify the dataloader works:
#   python Dataloader.py --root ./flickr8k
#   python Dataloader.py --root ./flickr8k --image_path ./flickr8k/Images/some_image.jpg

if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True,
                        help="Path to flickr8k/ folder")
    parser.add_argument("--image_path", type=str, default=None,
                        help="Optional: path to a specific image to display with its caption. "
                             "If not provided, displays the first image in the train batch.")
    args = parser.parse_args()

    train_loader, val_loader, test_loader, vocab = get_flickr8k_loaders(
        root_dir    = args.root,
        batch_size  = 32,
        num_workers = 0,    # 0 for quick local test; use 4+ on Discovery
    )

    print(f"\nVocabulary size : {len(vocab)}")
    print(f"Train batches   : {len(train_loader)}")
    print(f"Val   batches   : {len(val_loader)}")
    print(f"Test  batches   : {len(test_loader)}")

    images, captions, _ = next(iter(train_loader))
    print(f"\nBatch shapes")
    print(f"  images   : {tuple(images.shape)}")    # (32, 3, 224, 224)
    print(f"  captions : {tuple(captions.shape)}")  # (32, max_caption_length + 2)

    # ── Display image and decoded caption ─────────────────────────────
    if args.image_path:
        # Find this image in the training dataset and show its caption
        img_name  = os.path.basename(args.image_path)
        dataset   = train_loader.dataset
        match_idx = next(
            (i for i, name in enumerate(dataset.images) if name == img_name),
            None
        )

        raw_image = Image.open(args.image_path).convert("RGB")

        if match_idx is not None:
            _, caption_tensor, _ = dataset[match_idx]
            decoded = vocab.decode(caption_tensor.tolist())
            print(f"\nImage   : {args.image_path}")
            print(f"Caption : {decoded}")
        else:
            decoded = "(image not found in train split)"
            print(f"\nImage {img_name} not found in train split.")

    else:
        # Default: use first image from the batch
        decoded = vocab.decode(captions[0].tolist())
        print(f"\nDecoded first caption:\n  {decoded}")

        # Unnormalise for display (reverse the ImageNet normalisation)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        raw_image = (images[0] * std + mean).clamp(0, 1)
        raw_image = transforms.ToPILImage()(raw_image)

    plt.imshow(raw_image)
    plt.axis("off")
    plt.title(decoded)
    plt.show()