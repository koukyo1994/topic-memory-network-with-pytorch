import copy
import gensim
import numpy as np

from gensim.parsing.preprocessing import STOPWORDS

from scipy import sparse


def create_dictionary(msgs):
    dictionary = gensim.corpora.Dictionary(msgs)

    bow_dictionary = copy.deepcopy(dictionary)
    bow_dictionary.filter_tokens(
        list(map(bow_dictionary.token2id.get, STOPWORDS)))
    len_1_word = list(filter(lambda w: len(w) == 1, bow_dictionary.values()))
    bow_dictionary.filter_tokens(len_1_word)
    bow_dictionary.filter_extremes(no_below=3, keep_n=None)
    bow_dictionary.compactify()
    return dictionary, bow_dictionary


def get_wids(text_doc, seq_dictionary: gensim.corpora.Dictionary,
             bow_dictionary: gensim.corpora.Dictionary, ori_labels, logger):
    seq_doc = []
    row = []
    col = []
    value = []
    row_id = 0
    m_labels = []

    for d_i, doc in enumerate(text_doc):
        d2b_doc = bow_dictionary.doc2bow(doc)
        if len(d2b_doc) < 3:
            continue
        for i, j in d2b_doc:
            row.append(row_id)
            col.append(i)
            value.append(j)
        row_id += 1

        wids = list(map(seq_dictionary.token2id.get, doc))
        wids = np.array(list(filter(lambda x: x is not None, wids))) + 1
        m_labels.append(ori_labels[d_i])
        seq_doc.append(wids)
    lens = list(map(len, seq_doc))
    bow_doc = sparse.coo_matrix((value, (row, col)),
                                shape=(row_id, len(bow_dictionary))).tocsr()
    logger.info(
        f"get {len(seq_doc)} docs, avg len: {np.mean(lens)}, max len: {np.max(lens)}"
    )
    return seq_doc, bow_doc, m_labels
