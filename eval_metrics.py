from nltk.translate.bleu_score import corpus_bleu
import torch


def evaluation_metric(dataloader, hypotheses):
    """
    Compute BLEU-1 through BLEU-4 scores over the entire dataset.

    Parameters
    ----------
    ground_truth : list of list of list of str
        - All 5 reference captions per image, each tokenized into words.
        - outer list holds the inner lists and has size # images
        - next inner list is for each image and contains all references for that image (5 ref per image for flickr)
        - next inner list is for tokenized captions for an image 

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


# def prepare_hypotheses(all_generated, vocab):
#     hypotheses = []
#     for token_ids in all_generated:
#         words = vocab.decode(token_ids).split()
#         hypotheses.append(words)
#     return hypotheses

def test_model(model, data_loader, vocab, num_reference_per_image=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()

    all_generated = {}  # image_name -> generated caption (deduplicate by image)

    with torch.no_grad():
        for batch_data in data_loader:
            images, captions, image_names = batch_data

            assert all(image_names[j] == image_names[0] 
                for j in range(num_reference_per_image)), \
                    "First batch images aren't grouped by num_references_per_image — slicing will be wrong"

            
            images = images[::num_reference_per_image]

            # below are for debugging purposes
            captions = captions[::num_reference_per_image]
            image_names = image_names[::num_reference_per_image]

            images = images.to(device)

            batch_generated = model.generate(images, vocab)
            
            for img_name, prediction in zip(image_names, batch_generated):
                if img_name not in all_generated:
                    all_generated[img_name] = prediction

    # get references dict to ensure same ordering
    ref_dict = data_loader.dataset.get_all_references_dict()

    # # build hypotheses in same order as references
    hypotheses = [all_generated[img] for img in ref_dict.keys()]
    

    bleu_scores = evaluation_metric(data_loader, hypotheses)

    return bleu_scores

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