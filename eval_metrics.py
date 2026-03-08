# TODO come up with eval metric to use during training. BLEU?
def evaluation_metric(ground_truth, results):
    pass


# On BLEU-4: Yes, it's the standard metric for image captioning. The original Show and Tell paper reports BLEU-4, and it's still the most commonly cited score for comparing against other work. A couple of things worth knowing for your implementation:

# BLEU-4 compares n-grams (up to 4-grams) between your generated caption and the reference captions
# Flickr8k has 5 reference captions per image, and BLEU is designed to leverage multiple references — so make sure your evaluation_metric passes all 5 references, not just one. This is important because it significantly affects the score
# A reasonable baseline BLEU-4 on Flickr8k with a ResNet+LSTM is roughly 20–25