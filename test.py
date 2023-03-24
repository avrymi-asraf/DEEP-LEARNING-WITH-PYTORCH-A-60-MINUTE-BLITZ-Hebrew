import torch , torchtext
vectors = torchtext.vocab.FastText(language='he',vectors_cache=".vector_cache")
def find_nig(word, distans, maxRun):
    vec = vectors[word]
    res = []

    for new_word, ind in vectors.stoi.items():
        new_vec = vectors.vectors[ind]
        if torch.norm(vec - new_vec) < distans:
            res.append(new_word)
        maxRun -= 1
        if maxRun == 0:
            break

    return res

