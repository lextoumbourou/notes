def get_tf(document):
    output = {}
    for word in document.split():
        output[word] = output.get(word, 0) + 1

    return output


def get_idf(corpus):
    output = {}

    word_count = {}
    for document in corpus:
        document_word_count = {}

        words = document.split()
        for word in words:
            document_word_count[word] = 1

        word_count = {
            w: word_count.get(w, 0) + 1 for w in document_word_count.keys()}

    for word, count in word_count.item():
        output[word] = math.log(len(corpus) / (1 + count))

    return output[word]


def get_document_tf_idf(tf, idf):
    output = {}

    for w in tf:
        output[w] = tf[w] * idf[w]

    return output


def calculate_similarities(tf_idf_1, tf_idf_2):
    output = {}

    for w in tf_idf_1:
        output[w] = tf_idf_1[w] * tf_idf_2[w]
