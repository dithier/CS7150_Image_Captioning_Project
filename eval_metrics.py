"""
# TODO come up with eval metric to use during training. BLEU?
def evaluation_metric(ground_truth, results):
    pass


# On BLEU-4: Yes, it's the standard metric for image captioning. The original Show and Tell paper reports BLEU-4, and it's still the most commonly cited score for comparing against other work. A couple of things worth knowing for your implementation:

# BLEU-4 compares n-grams (up to 4-grams) between your generated caption and the reference captions
# Flickr8k has 5 reference captions per image, and BLEU is designed to leverage multiple references — so make sure your evaluation_metric passes all 5 references, not just one. This is important because it significantly affects the score
# A reasonable baseline BLEU-4 on Flickr8k with a ResNet+LSTM is roughly 20–25
"""

"""
Evaluation Metrics for Image Captioning
========================================
Implements BLEU-1 through BLEU-4 using nltk's corpus_bleu.

Why corpus_bleu and not sentence_bleu?
    corpus_bleu computes BLEU over the entire dataset at once rather than
    averaging per-sentence scores. This is the standard for image captioning
    and matches how the original SAT paper reports results.

Why all 5 reference captions?
    Flickr8k has 5 human-written captions per image. BLEU is designed to
    leverage multiple references — passing all 5 significantly improves the
    score and gives a fairer evaluation.

Expected input format:
    ground_truth : list of list of list of str
        Outer list  : one entry per image
        Middle list : one entry per reference caption (5 for Flickr8k)
        Inner list  : tokenized words of that caption
        e.g. [[["a", "dog", "runs"], ["brown", "dog", "plays"]], ...]

    hypotheses : list of list of str
        One tokenized generated caption per image
        e.g. [["a", "dog", "runs"], ...]

Install dependencies:
    pip install nltk
"""

from nltk.translate.bleu_score import corpus_bleu

def evaluation_metric(dataloader, hypotheses):
    """
    Compute BLEU-1 through BLEU-4 scores over the entire dataset.

    Parameters
    ----------
    ground_truth : list of list of list of str
        All 5 reference captions per image, each tokenized into words.
        outer list holds the inner lists
        next inner list is for each image
        next inner list is for each caption for an image (will be 5 items for flickr)

    hypotheses : list of list of str
        One generated caption per image, tokenized into words.
        one list per image
        inner list is length of caption (tokenized words)

    Returns
    -------
    dict with keys 'bleu1', 'bleu2', 'bleu3', 'bleu4'
    Values are floats between 0 and 1 (multiply by 100 for percentage).
    """
    ground_truth = dataloader.dataset.get_all_references()

    # BLEU-1: only unigrams (individual words)
    bleu1 = corpus_bleu(ground_truth, hypotheses, weights=(1.0, 0, 0, 0))

    # BLEU-2: unigrams + bigrams (pairs of consecutive words)
    bleu2 = corpus_bleu(ground_truth, hypotheses, weights=(0.5, 0.5, 0, 0))

    # BLEU-3: unigrams + bigrams + trigrams
    bleu3 = corpus_bleu(ground_truth, hypotheses, weights=(0.33, 0.33, 0.33, 0))

    # BLEU-4: unigrams through 4-grams — the standard metric for image captioning
    # SAT paper baseline on Flickr8k with ResNet+LSTM is roughly 20-25 BLEU-4
    bleu4 = corpus_bleu(ground_truth, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))

    return {
        "bleu1": bleu1,
        "bleu2": bleu2,
        "bleu3": bleu3,
        "bleu4": bleu4,
    }


# def prepare_references_and_hypotheses(all_references, all_generated, vocab):
#     """
#     Convert raw token ID tensors from the dataloader and model into the
#     string format expected by corpus_bleu.

#     Parameters
#     ----------
#     all_references : list of tensors  (5, max_caption_length+1) per image
#         The ground truth caption tensors from the dataloader.

#     all_generated : list of list of int
#         The generated caption token IDs from model.generate().

#     vocab : Vocabulary
#         The vocabulary object used to decode token IDs to words.

#     Returns
#     -------
#     references  : list of list of list of str  — for corpus_bleu
#     hypotheses  : list of list of str          — for corpus_bleu
#     """
#     references = []
#     hypotheses = []

#     for ref_tensor, gen_ids in zip(all_references, all_generated):
#         # Decode all 5 reference captions for this image
#         # Each is tokenized into a list of words (special tokens stripped by decode)
#         image_refs = [
#             vocab.decode(ref_tensor[i].tolist()).split()
#             for i in range(ref_tensor.size(0))  # iterate over 5 captions
#         ]
#         references.append(image_refs)

#         # Decode the generated caption into a list of words
#         hypothesis = vocab.decode(gen_ids).split()
#         hypotheses.append(hypothesis)

#     return references, hypotheses