text = """
Word2vec is a technique in natural language processing (NLP) for obtaining vector representations of words. These vectors capture information about the meaning of the word based on the surrounding words. The word2vec algorithm estimates these representations by modeling text in a large corpus. Once trained, such a model can detect synonymous words or suggest additional words for a partial sentence. Word2vec was developed by Tomáš Mikolov and colleagues at Google and published in 2013.

Word2vec represents a word as a high-dimension vector of numbers which capture relationships between words. In particular, words which appear in similar contexts are mapped to vectors which are nearby as measured by cosine similarity. This indicates the level of semantic similarity between the words, so for example the vectors for walk and ran are nearby, as are those for but and however and Berlin and Germany.

Word2vec is a group of related models that are used to produce word embeddings. These models are shallow, two-layer neural networks that are trained to reconstruct linguistic contexts of words. Word2vec takes as its input a large corpus of text and produces a vector space, typically of several hundred dimensions, with each unique word in the corpus being assigned a corresponding vector in the space.

Word2vec can utilize either of two model architectures to produce these distributed representations of words: Continuous Bag-Of-Words (CBOW) or continuously sliding skip-gram. In both architectures, word2vec considers both individual words and a sliding context window as it iterates over the corpus.

The CBOW can be viewed as a fill in the blank task, where the word embedding represents the way the word influences the relative probabilities of other words in the context window. Words which are semantically similar should influence these probabilities in similar ways, because semantically similar words should be used in similar contexts. The order of context words does not influence prediction (bag-of-words assumption).

In the continuous skip-gram architecture, the model uses the current word to predict the surrounding window of context words. The skip-gram architecture weighs nearby context words more heavily than more distant context words. According to the authors' note, CBOW is faster while skip-gram does a better job for infrequent words.

After the model has trained, the learned word embeddings are positioned in the vector space such that words that share common contexts in the corpus — that is, words that are semantically and syntactically similar — are located close to one another in the space. More dissimilar words are located farther from one another in the space.

This section is based on expositions.

A corpus is a sequence of words. Both CBOW and skip-gram are methods to learn one vector per word appearing in the corpus.

The idea of skip-gram is that the vector of a word should be close to the vector of each of its neighbors. The idea of CBOW is that the vector-sum of a word's neighbors should be close to the vector of the word.

In the original publication, closeness is measured by softmax, but the framework allows other ways to measure closeness.

Continuous Bag of Words (CBOW)
Suppose we want each word in the corpus to be predicted by every other word in a small span of 4 words.

Then the training objective is to maximize the following quantity:

That is, we want to maximize the total probability for the corpus, as seen by a probability model that uses word neighbors to predict words.
Products are numerically unstable, so we convert it by taking the logarithm:

That is, we maximize the log-probability of the corpus.
Our probability model is as follows: Given words 

The quantity to be maximized is then after simplifications:

The quantity on the left is fast to compute, but the quantity on the right is slow, as it involves summing over the entire vocabulary set for each word in the corpus. Furthermore, to use gradient ascent to maximize the log-probability requires computing the gradient of the quantity on the right, which is intractable. This prompted the authors to use numerical approximation tricks.
Skip-gram
For skip-gram, the training objective is

That is, we want to maximize the total probability for the corpus, as seen by a probability model that uses words to predict its word neighbors. We predict each word-neighbor independently, thus 

There is only a single difference from the CBOW equation, highlighted in red.

In 2010, Tomáš Mikolov (then at Brno University of Technology) with co-authors applied a simple recurrent neural network with a single hidden layer to language modelling.

Word2vec was created, patented, and published in 2013 by a team of researchers led by Mikolov at Google over two papers. Other researchers helped analyse and explain the algorithm. Embedding vectors created using the Word2vec algorithm have some advantages compared to earlier algorithms such as those using n-grams and latent semantic analysis.

By 2022, the straight Word2vec approach was described as "dated." Transformer models, which add multiple neural-network attention layers on top of a word embedding model similar to Word2vec, have come to be regarded as the state of the art in NLP.

Results of word2vec training can be sensitive to parametrization. The following are some important parameters in word2vec training.

A Word2vec model can be trained with hierarchical softmax and/or negative sampling. To approximate the conditional log-likelihood a model seeks to maximize, the hierarchical softmax method uses a Huffman tree to reduce calculation. The negative sampling method, on the other hand, approaches the maximization problem by minimizing the log-likelihood of sampled negative instances. According to the authors, hierarchical softmax works better for infrequent words while negative sampling works better for frequent words and better with low dimensional vectors. As training epochs increase, hierarchical softmax stops being useful.

High-frequency and low-frequency words often provide little information. Words with a frequency above a certain threshold, or below a certain threshold, may be subsampled or removed to speed up training.

Quality of word embedding increases with higher dimensionality. But after reaching some point, marginal gain diminishes. Typically, the dimensionality of the vectors is set to be between 100 and 1,000.
The size of the context window determines how many words before and after a given word are included as context words of the given word. According to the authors' note, the recommended value is 10 for skip-gram and 5 for CBOW.

There are a variety of extensions to word2vec.

doc2vec, generates distributed representations of variable-length pieces of texts, such as sentences, paragraphs, or entire documents. doc2vec has been implemented in the C, Python and Java/Scala tools (see below), with the Java and Python versions also supporting inference of document embeddings on new, unseen documents.

doc2vec estimates the distributed representations of documents much like how word2vec estimates representations of words: doc2vec utilizes either of two model architectures, both of which are allegories to the architectures used in word2vec. The first, Distributed Memory Model of Paragraph Vectors (PV-DM), is identical to CBOW other than it also provides a unique document identifier as a piece of additional context. The second architecture, Distributed Bag of Words version of Paragraph Vector (PV-DBOW), is identical to the skip-gram model except that it attempts to predict the window of surrounding context words from the paragraph identifier instead of the current word.

doc2vec also has the ability to capture the semantic meanings for additional pieces of  context around words; doc2vec can estimate the semantic embeddings for speakers or speaker attributes, groups, and periods of time. For example, doc2vec has been used to estimate the political positions of political parties in various Congresses and Parliaments in the U.S. and U.K., respectively, and various governmental institutions.

Another extension of word2vec is top2vec, which leverages both document and word embeddings to estimate distributed representations of topics. top2vec takes document embeddings learned from a doc2vec model and reduces them into a lower dimension (typically using UMAP). The space of documents is then scanned using HDBSCAN, and clusters of similar documents are found. Next, the centroid of documents identified in a cluster is considered to be that cluster's topic vector. Finally, top2vec searches the semantic space for word embeddings located near to the topic vector to ascertain the 'meaning' of the topic. The word with embeddings most similar to the topic vector might be assigned as the topic's title, whereas far away word embeddings may be considered unrelated.

As opposed to other topic models such as LDA, top2vec provides canonical distance metrics between two topics, or between a topic and another embeddings (word, document, or otherwise). Together with results from HDBSCAN, users can generate topic hierarchies, or groups of related topics and subtopics.

Furthermore, a user can use the results of top2vec to infer the topics of out-of-sample documents. After inferring the embedding for a new document, must only search the space of topics for the closest topic vector.

An extension of word vectors for n-grams in biological sequences (e.g. DNA, RNA, and proteins) for bioinformatics applications has been proposed by Asgari and Mofrad. Named bio-vectors (BioVec) to refer to biological sequences in general with protein-vectors (ProtVec) for proteins (amino-acid sequences) and gene-vectors (GeneVec) for gene sequences, this representation can be widely used in applications of machine learning in proteomics and genomics. The results suggest that BioVectors can characterize biological sequences in terms of biochemical and biophysical interpretations of the underlying patterns. A similar variant, dna2vec, has shown that there is correlation between Needleman–Wunsch similarity score and cosine similarity of dna2vec word vectors.

Radiology and intelligent word embeddings
An extension of word vectors for creating a dense vector representation of unstructured radiology reports has been proposed by Banerjee et al. One of the biggest challenges with Word2vec is how to handle unknown or out-of-vocabulary (OOV) words and morphologically similar words. If the Word2vec model has not encountered a particular word before, it will be forced to use a random vector, which is generally far from its ideal representation. This can particularly be an issue in domains like medicine where synonyms and related words can be used depending on the preferred style of radiologist, and words may have been used infrequently in a large corpus.

IWE combines Word2vec with a semantic dictionary mapping technique to tackle the major challenges of information extraction from clinical texts, which include ambiguity of free text narrative style, lexical variations, use of ungrammatical and telegraphic phases, arbitrary ordering of words, and frequent appearance of abbreviations and acronyms. Of particular interest, the IWE model (trained on the one institutional dataset) successfully translated to a different institutional dataset which demonstrates good generalizability of the approach across institutions.

The reasons for successful word embedding learning in the word2vec framework are poorly understood. Goldberg and Levy point out that the word2vec objective function causes words that occur in similar contexts to have similar embeddings (as measured by cosine similarity) and note that this is in line with J. R. Firth's distributional hypothesis. However, they note that this explanation is "very hand-wavy" and argue that a more formal explanation would be preferable.

Levy shows that much of the superior performance of word2vec or similar embeddings in downstream tasks is not a result of the models per se, but of the choice of specific hyperparameters. Transferring these hyperparameters to more 'traditional' approaches yields similar performances in downstream tasks. Arora explains word2vec and related algorithms as performing inference for a simple generative model for text, which involves a random walk generation process based upon loglinear topic model. They use this to explain some properties of word embeddings, including their use to solve analogies.

Preservation of semantic and syntactic relationships
Visual illustration of word embeddings
The word embedding approach is able to capture multiple different degrees of similarity between words. Mikolov found that semantic and syntactic patterns can be reproduced using vector arithmetic. Patterns such as "Man is to Woman as Brother is to Sister" can be generated through algebraic operations on the vector representations of these words such that the vector representation of "Brother" - "Man" + "Woman" produces a result which is closest to the vector representation of "Sister" in the model. Such relationships can be generated for a range of semantic relations (such as Country–Capital) as well as syntactic relations (e.g. present tense–past tense).

This facet of word2vec has been exploited in a variety of other contexts. For example, word2vec has been used to map a vector space of words in one language to a vector space constructed from another language. Relationships between translated words in both spaces can be used to assist with machine translation of new words.

Assessing the quality of a model
Mikolov et al. (2013) develop an approach to assessing the quality of a word2vec model which draws on the semantic and syntactic patterns discussed above. They developed a set of 8,869 semantic relations and 10,675 syntactic relations which they use as a benchmark to test the accuracy of a model. When assessing the quality of a vector model, a user may draw on this accuracy test which is implemented in word2vec, or develop their own test set which is meaningful to the corpora which make up the model. This approach offers a more challenging test than simply arguing that the words most similar to a given test word are intuitively plausible.

Parameters and model quality
The use of different model parameters and different corpus sizes can greatly affect the quality of a word2vec model. Accuracy can be improved in a number of ways, including the choice of model architecture (CBOW or Skip-Gram), increasing the training data set, increasing the number of vector dimensions, and increasing the window size of words considered by the algorithm. Each of these improvements comes with the cost of increased computational complexity and therefore increased model generation time.

In models using large corpora and a high number of dimensions, the skip-gram model yields the highest overall accuracy, and consistently produces the highest accuracy on semantic relationships, as well as yielding the highest syntactic accuracy in most cases. However, the CBOW is less computationally expensive and yields similar accuracy results.

Overall, accuracy increases with the number of words used and the number of dimensions. Mikolov et al. report that doubling the amount of training data results in an increase in computational complexity equivalent to doubling the number of vector dimensions.

Altszyler and coauthors studied Word2vec performance in two semantic tests for different corpus size. They found that Word2vec has a steep learning curve, outperforming another word-embedding technique, latent semantic analysis (LSA), when it is trained with medium to large corpus size (more than 10 million words). However, with a small training corpus, LSA showed better performance. Additionally they show that the best parameter setting depends on the task and the training corpus. Nevertheless, for skip-gram models trained in medium size corpora, with 50 dimensions, a window size of 15 and 10 negative samples seems to be a good parameter setting.

"""

def process_text(context_size=2, text=text):
    
    #pad text with context window
    for i in range(0, context_size):
        text = "<pad> " + text + " <pad>"

    text = text.split()

    vocab = set(text)
    vocab_size = len(vocab)

    word_to_ix = {word:ix for ix, word in enumerate(vocab)}
    ix_to_word = {ix:word for ix, word in enumerate(vocab)}

    data = []

    for i in range(context_size, len(text) - context_size):
        context_i = []
        for j in range(1, context_size+1):
            context_i.append(text[i-j])
            context_i.append(text[i+j])

        target = text[i]
        data.append((context_i, target))
    return data, word_to_ix, ix_to_word, vocab_size
