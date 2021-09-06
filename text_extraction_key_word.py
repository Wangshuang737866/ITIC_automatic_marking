from nltk import tokenize
from operator import itemgetter
import math
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import argparse
import csv
import editdistance
import numpy as np
from ngram import NGram
#print("td_idf_score",tf_idf_score)
import jaro
import re
import math
from collections import Counter
import soundex
stop_words = set(stopwords.words('english'))
class text_extraction_key_word:

    # print("stop_words",stop_words)
    # print("total_w_l",total_word_length)

    def get_top_n(self,dict_elem, n):
        result = dict(sorted(dict_elem.items(), key=itemgetter(1), reverse=True)[:n])
        return result

    # print("tf_score",tf_score)
    def check_sent(self,word, sentences):
        final = [all([w in x for w in word]) for x in sentences]
        sent_len = [sentences[i] for i in range(0, len(final)) if final[i]]
        return int(len(sent_len))

    def caculate_tf_score(self,str):
        total_words = str.split()
        total_sentences = tokenize.sent_tokenize(str)
        total_sent_len = len(total_sentences)
        total_word_length = len(total_words)
        tf_score = {}
        for each_word in total_words:
            each_word = each_word.replace('.', '')
            # print(each_word)
            if each_word not in stop_words:
                if each_word in tf_score:
                    tf_score[each_word] += 1
                else:
                    tf_score[each_word] = 1
        # print(tf_score)
        # Dividing by total_word_length for each dictionary element
        tf_score.update((x, y / int(total_word_length)) for x, y in tf_score.items())
        return tf_score

    def caculate_IDF_score(self,str):
        total_words = str.split()
        total_sentences = tokenize.sent_tokenize(str)
        total_sent_len = len(total_sentences)
        idf_score = {}
        for each_word in total_words:
            each_word = each_word.replace('.', '')
            if each_word not in stop_words:
                if each_word in idf_score:
                    idf_score[each_word] = self.check_sent(each_word, total_sentences)
                else:
                    idf_score[each_word] = 1
        # print(idf_score)
        # Performing a log and divide
        idf_score.update((x, math.log(int(total_sent_len) / y)) for x, y in idf_score.items())
        return idf_score

    def calculate_tf_IDF_score(self,tf_score, idf_score):
        tf_idf_score = {key: tf_score[key] * idf_score.get(key, 0) for key in tf_score.keys()}
        return tf_idf_score

    def calculate_editdistance_score(self,total_words, n):
        ed_matrix = np.zeros((len(total_words), len(total_words)))
        # ed = editdistance.distance(total_words[0], total_words[1])
        for i in range(len(total_words)):
            for j in range(len(total_words)):
                ed_matrix[i][j] = editdistance.distance(total_words[i], total_words[j])
        score = np.sum(ed_matrix, axis=0)
        # print(score)
        score0 = np.sort(score)
        # print(score0)
        index = np.argsort(score)
        # print(index)
        score1 = []
        keywords1 = []
        for i in range(n):
            # print(index[i])
            # t=index[i]
            # print(t)
            score1.append(score0[i])
            # print(score1)
            keywords1.append(total_words[index[i]])
            # print(keywords1)

        return score0[0:n] / sum(score0[0:n]), keywords1

    def calculate_Jaro_score(self,total_words, n):
        ed_matrix = np.zeros((len(total_words), len(total_words)))
        # ed = editdistance.distance(total_words[0], total_words[1])
        for i in range(len(total_words)):
            for j in range(len(total_words)):
                ed_matrix[i][j] = jaro.jaro_metric(total_words[i], total_words[j])
        score = np.sum(ed_matrix, axis=0)
        # print(score)
        score0 = np.sort(-score)
        # print(score0)
        index = np.argsort(-score)
        # print(index)
        score1 = []
        keywords1 = []
        for i in range(n):
            # print(index[i])
            # t=index[i]
            # print(t)
            score1.append(score[index[i]])
            # print(score1)
            keywords1.append(total_words[index[i]])
            # print(keywords1)

        return -score0[0:n] / sum(-score0[0:n]), keywords1

    def calculate_NGram_score(self,total_words, n):
        NG_matrix = np.zeros((len(total_words), len(total_words)))
        # ed = editdistance.distance(total_words[0], total_words[1])
        for i in range(len(total_words)):
            for j in range(len(total_words)):
                NG_matrix[i][j] = NGram.compare(total_words[i], total_words[j])
        score = np.sum(NG_matrix, axis=0)
        # print(score)
        score0 = np.sort(-score)
        # print(score0)
        index = np.argsort(-score)
        # print(index)
        score1 = []
        keywords1 = []
        for i in range(n):
            # print(index[i])
            # t=index[i]
            # print(t)
            score1.append(score[index[i]])
            # print(score1)
            keywords1.append(total_words[index[i]])
            # print(keywords1)

        return -score0[0:n] / sum(-score0[0:n]), keywords1

    def jaccard(self,list1, list2):
        intersection = len(list(set(list1).intersection(list2)))
        union = (len(list1) + len(list2)) - intersection
        return float(intersection) / union

    def calculate_jaccard_score(self,total_words, n):
        Jac_matrix = np.zeros((len(total_words), len(total_words)))
        # ed = editdistance.distance(total_words[0], total_words[1])
        for i in range(len(total_words)):
            for j in range(len(total_words)):
                Jac_matrix[i][j] = self.jaccard(list(total_words[i]), list(total_words[j]))
        score = np.sum(Jac_matrix, axis=0)
        # print(score)
        score0 = np.sort(-score)
        # print(score0)
        index = np.argsort(-score)
        # print(index)
        score1 = []
        keywords1 = []
        for i in range(n):
            # print(index[i])
            # t=index[i]
            # print(t)
            score1.append(score[index[i]])
            # print(score1)
            keywords1.append(total_words[index[i]])
            # print(keywords1)

        return -score0[0:n] / sum(-score0[0:n]), keywords1

    def get_cosine(self,vec1, vec2):
        intersection = set(vec1.keys()) & set(vec2.keys())
        numerator = sum([vec1[x] * vec2[x] for x in intersection])

        sum1 = sum([vec1[x] ** 2 for x in vec1.keys()])
        sum2 = sum([vec2[x] ** 2 for x in vec2.keys()])
        denominator = math.sqrt(sum1) * math.sqrt(sum2)

        if not denominator:
            return 0.0
        else:
            return float(numerator) / denominator

    def text_to_vector(self,text):
        word = re.compile(r'\w+')
        words = word.findall(text)
        return Counter(words)

    def get_result(self,content_a, content_b):
        text1 = content_a
        text2 = content_b

        vector1 = self.text_to_vector(text1)
        vector2 = self.text_to_vector(text2)

        cosine_result = self.get_cosine(vector1, vector2)
        return cosine_result

    def calculate_cosine_score(self,total_words, n):
        Jac_matrix = np.zeros((len(total_words), len(total_words)))
        # ed = editdistance.distance(total_words[0], total_words[1])
        for i in range(len(total_words)):
            for j in range(len(total_words)):
                # print(type(total_words[i]))
                # count_a=text_to_vector(total_words[i])
                # count_b=text_to_vector(total_words[i])
                Jac_matrix[i][j] = self.get_result(total_words[i], total_words[j])
        score = np.sum(Jac_matrix, axis=0)
        # print(score)
        score0 = np.sort(-score)
        # print(score0)
        index = np.argsort(-score)
        # print(index)
        score1 = []
        keywords1 = []
        for i in range(n):
            # print(index[i])
            # t=index[i]
            # print(t)
            score1.append(score[index[i]])
            # print(score1)
            keywords1.append(total_words[index[i]])
            # print(keywords1)

        return -score0[0:n] / sum(-score0[0:n]), keywords1

    def calculate_soundex_score(self,total_words, n):
        Jac_matrix = np.zeros((len(total_words), len(total_words)))
        # ed = editdistance.distance(total_words[0], total_words[1])
        for i in range(len(total_words)):
            for j in range(len(total_words)):
                # print(type(total_words[i]))
                # count_a=text_to_vector(total_words[i])
                # count_b=text_to_vector(total_words[i])
                Jac_matrix[i][j] = soundex.g(total_words[i], j)
                print(Jac_matrix[i][j])
        score = np.sum(Jac_matrix, axis=0)
        # print(score)
        score0 = np.sort(-score)
        # print(score0)
        index = np.argsort(-score)
        # print(index)
        score1 = []
        keywords1 = []
        for i in range(n):
            # print(index[i])
            # t=index[i]
            # print(t)
            score1.append(score[index[i]])
            # print(score1)
            keywords1.append(total_words[index[i]])
            # print(keywords1)

        return -score0[0:n] / sum(-score0[0:n]), keywords1

if __name__ == '__main__':
        #   parser = argparse.ArgumentParser()
        #   parser.add_argument('--n', type=int, default=5)
        #   parser.add_argument('--dataset', type=str, default='answer', help="")
        #   parser.add_argument('--data_path', type=str, default='student_answer')
        # #parser.add_argument('--dataname', type=str, default='answer', help="")
        #   args = parser.parse_args()
        #   f = open(args.data_path)
        #   doc = f.read()
        # doc = 'I am a graduate. I want to learn Python. I like learning Python. Python is easy. Python is interesting. ' \
        # 'Learning increases thinking. Everyone should invest time in learning'
        batch_size = 5;
        csv_reader = csv.reader(open("data_essay.csv"))
        data = []
        sentence_num = []
        paragraphs = []
        for line in csv_reader:
            # print(line[1])
            data.append(line[1])
            paragraph = "".join(line[1])
            paragraphs.append(paragraph)
        doc = paragraphs[0]
        total_words = doc.split()
        total_word_length = len(total_words)
        # print(total_word_length)
        stop_words = set(stopwords.words('english'))
        # print("stop_words",stop_words)
        # print("total_w_l",total_word_length)
        total_sentences = tokenize.sent_tokenize(doc)
        total_sent_len = len(total_sentences)
        # print("total_sent_len",total_sent_len)
        # print(total_words)
        tekw=text_extraction_key_word()
        tf_score = tekw.caculate_tf_score(doc)

        #   tf_score = {}
        #   for each_word in total_words:
        #       each_word = each_word.replace('.', '')
        #       #print(each_word)
        #       if each_word not in stop_words:
        #           if each_word in tf_score:
        #               tf_score[each_word] += 1
        #           else:
        #               tf_score[each_word] = 1
        #  # print(tf_score)
        # # Dividing by total_word_length for each dictionary element
        #   tf_score.update((x, y / int(total_word_length)) for x, y in tf_score.items())
        #   idf_score = {}
        #   for each_word in total_words:
        #       each_word = each_word.replace('.', '')
        #       if each_word not in stop_words:
        #           if each_word in idf_score:
        #               idf_score[each_word] = check_sent(each_word, total_sentences)
        #           else:
        #               idf_score[each_word] = 1
        #   #print(idf_score)
        # # Performing a log and divide
        #   idf_score.update((x, math.log(int(total_sent_len) / y)) for x, y in idf_score.items())
        idf_score = tekw.caculate_IDF_score(doc)
        # print("idf_score",idf_score)
        tf_idf_score = tekw.calculate_tf_IDF_score(tf_score, idf_score)
        # tf_idf_score = {key: tf_score[key] * idf_score.get(key, 0) for key in tf_score.keys()}
        # print(tf_idf_score.values())
        #
        # n=5
        # n=args.__init__()
        # print(str)
        keywords = tekw.get_top_n(tf_idf_score, batch_size)
        print("keywords_TF_IDF", keywords)
        # ed=editdistance.distance(total_words[0],total_words[1])
        # print(ed)
        total_words_unique = sorted(set(total_words))
        print(total_words_unique)
        ed_score, keywords_ed = tekw.calculate_editdistance_score(total_words_unique, batch_size)
        # print("ed_matrix",ed_score)
        print("keywords_ed", keywords_ed)
        jaro_score, keyword_jaro = tekw.calculate_Jaro_score(total_words_unique, batch_size)
        print("Jaro_matrix", jaro_score)
        print("keywords_Jaro", keyword_jaro)
        Ng_score, keyword_Ng = tekw.calculate_NGram_score(total_words_unique, batch_size)
        print("Ng_matrix", Ng_score)
        print("keywords_Ng", keyword_Ng)
        jaccard_score, keyword_jaccard = tekw.calculate_jaccard_score(total_words_unique, batch_size)
        print("jaccard_matrix", jaccard_score)
        print("keywords_jaccard", keyword_jaccard)
        total_words1 = doc.split(".")
        print(total_words1[1])
        cosine_score, keyword_cosine = tekw.calculate_cosine_score(total_words_unique, batch_size)
        print("cosine_matrix", cosine_score)
        print("keywords_cosine", keyword_cosine)
        # soundex_score, keyword_soundex = calculate_soundex_score(total_words_unique, batch_size)
        # print("cosine_matrix", soundex_score)
        # print("keywords_cosine", keyword_soundex)