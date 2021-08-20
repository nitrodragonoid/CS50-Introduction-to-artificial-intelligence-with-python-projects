import nltk
import sys
import os
import string
import math
import gensim

#nltk.download('stopwords')
#python3 -m nltk.downloader stopwords

FILE_MATCHES = 1
SENTENCE_MATCHES = 1
corpus = 'manga_corpus'


def main():

    # Check command-line arguments
    #if len(sys.argv) != 2:
    #    sys.exit("Usage: python3 questions.py corpus")

    # Calculate IDF values across files
    files = load_files(corpus)
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    for i in range(10):
    # Prompt user for query
        query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
        filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
        sentences = dict()
        for filename in filenames:
            for passage in files[filename].split("\n"):
                for sentence in nltk.sent_tokenize(passage):
                    tokens = tokenize(sentence)
                #tokens = gensim.utils.simple_preprocess(sentence, deacc=True) 
                    if tokens:
                        sentences[sentence] = tokens

    # Compute IDF values across sentences
        idfs = compute_idfs(sentences)

    # Determine top sentence matches
        matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
        for match in matches:
            print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    text_dict = {}
    for folder,_, all_files in os.walk(directory):
        for file in all_files:
            print(file)
            with open(os.path.join(folder, file),'r') as f:
                text_dict[file] = f.read()
    return text_dict
    #raise NotImplementedError


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    punctuations = string.punctuation
    stop_words = nltk.corpus.stopwords.words("english")
    #stop_words = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]
    words_list = nltk.word_tokenize(document.lower())
    #words_list = gensim.utils.simple_preprocess(document.lower(), deacc=True) 
    words = [w for w in words_list if w not in punctuations and w not in stop_words]
    return words
    #raise NotImplementedError


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    idf_dict = {}
    n = len(documents)
    words = set(w for lst in documents.values() for w in lst)
    for i in words:
        a = 0
        for doc in documents.values():
            if i in doc:
                a += 1
        idf = math.log(n/a)
        #idf = float(n/a)
        idf_dict[i] = idf
    return idf_dict
    #raise NotImplementedError


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    score = {}
    for name in files:
        words = files[name]
        tf_idf = 0
        for q in query:
            tf_idf += words.count(q) * idfs[q]
        score[name] = tf_idf
    
    filenames  = sorted(score.items(), key=lambda x: x[1], reverse=True)
    filenames = [x[0] for x in filenames]

    return filenames[:n]

    #raise NotImplementedError


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    scores = {}
    for sent, words in sentences.items():
        common_words = query.intersection(words)
        val = 0
        for w in common_words:
            val += idfs[w]

        num_query = sum(map(lambda x: x in common_words, words))
        query_term_density = num_query / len(words)

        scores[sent] = {'idf': val, 'qtd': query_term_density}

    ranked_sentences = sorted(scores.items(), key=lambda x: (x[1]['idf'], x[1]['qtd']), reverse=True)
    ranked_sentences = [x[0] for x in ranked_sentences]

    return ranked_sentences[:n]

    #raise NotImplementedError


if __name__ == "__main__":
    main()
