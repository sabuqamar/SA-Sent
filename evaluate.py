import tokenizer
import _pickle as cPickle 
from model import *
from config import config
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

def get_vector(text):
    inputs = tokenizer(text, return_tensors='pt')
    outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].detach().numpy()

def get_score(label):
    if label == 0:
        return 1
    elif label == 2:
        return -1
    return 0
def get_clusters(terms, labels):
    categories = ["service", "food", "price", "ambience"]
    categories_scores = [0, 0, 0, 0]
    category_vectors = [get_vector(cat) for cat in categories]
    word_vectors = [get_vector(word) for word in terms]

    similarity_threshold = 0.7
    category_assignments = []

    for idx, word_vec in enumerate(word_vectors):
        similarities = [cosine_similarity(word_vec, cat_vec) for cat_vec in category_vectors]
        max_similarity = max(similarities)
        if max_similarity >= similarity_threshold:
            category_index = similarities.index(max_similarity)
            category_assignments.append(categories[category_index])
            print(labels[idx])
            categories_scores[category_index] += get_score(labels[idx])
        else:
            category_assignments.append(None)
    return categories_scores

def get_sent_mask(sent, terms):
	with open("word2id", "rb") as f:
		word2id = cPickle.load(f)
	sent_tokens = tokenize(sent)
	sent_inds = [word2id[x] if x in word2id else word2id["UNK"] 
                for x in sent_tokens]
	masks = []
	for term in terms:
		target_tokens = tokenize(term)
		try:
			target_start = sent_tokens.index(target_tokens[0])
			target_end = sent_tokens[max(0, target_start - 1):].index(target_tokens[-1])  + max(0, target_start - 1)
		except:
			continue
		mask = [0] * len(sent_tokens)
		for m_i in range(target_start, target_end + 1):
			mask[m_i] = 1
		masks.append(mask)
	return sent_inds, masks

def evaluate(sent_inds, masks):
	with open("id2word", "rb") as f:
		id2word = cPickle.load(f)
	model = torch.load("model_params.json")
	model.eval()

	print("transitions matrix ", model.inter_crf.transitions.data)
	# Initialize empty lists to store true labels and predicted labels
	terms = []
	labels = []
	for mask in masks:
		pred_label, best_seq = model.predict(sent_inds, mask)
		term = []
		acc_sent = []
		for i, x in enumerate(sent_inds):
			acc_sent.append(id2word[x])
			if mask[i] == 1:
				term.append(id2word[x])
		terms.append(u" ".join(term))
		labels.append(pred_label)
	print("scores: ", get_clusters(terms, labels))

def tokenize(sent_str):
        sent_str = " ".join(sent_str.split("-"))
        sent_str = " ".join(sent_str.split("/"))
        sent_str = " ".join(sent_str.split("!"))
        return tokenizer.tokenize(sent_str)

if __name__ == "__main__":
	sent = "All the money went into the interior decoration, none of it went to the chefs"
	terms = ["interior decoration"]
	sent_inds, masks = get_sent_mask(sent, terms)
	evaluate(sent_inds, masks)

