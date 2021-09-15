from nltk import tokenize
from operator import itemgetter
import math
import nltk
from nltk.corpus import stopwords
import re
import math
import string
stop_words = set(stopwords.words('english'))
class Math_rubric:
    def __init__(self, qualifiedanswer=None, rubric=None, total_mark=None):

        self.qualifiedanswer = qualifiedanswer
        self.rubric = rubric
        self.total_mark =total_mark


    def get_top_n(self, dict_elem, n):
        result = dict(sorted(dict_elem.items(), key = itemgetter(1), reverse = True)[:n])
        return result
    # print("tf_score",tf_score)
    def check_sent(self, word, sentences):
         final = [all([w in x for w in word]) for x in sentences]
         sent_len = [sentences[i] for i in range(0, len(final)) if final[i]]
         return int(len(sent_len))
    def caculate_tf_score(self,total_words,stop_words,total_word_length):
        tf_score = {}
        for each_word in total_words:
            each_word = each_word.replace('.', '')
            # print(each_word)
            if each_word not in stop_words:
                #print("each_word",each_word)
                if each_word in tf_score:
                    s1=re.compile(r'\d+')
                    s=s1.findall(each_word)
                    if len(s)>0:
                      tf_score[each_word] += 5
                    else:
                        tf_score[each_word] += 1
                else:
                    s1 = re.compile(r'\d+')
                    s = s1.findall(each_word)
                    if len(s)>0:
                        tf_score[each_word] = 5
                    else:
                        tf_score[each_word] = 1
        # print(tf_score)
        # Dividing by total_word_length for each dictionary element
        tf_score.update((x, y / int(total_word_length)) for x, y in tf_score.items())
        return tf_score
    def caculate_IDF_score(self, total_words,stop_words,total_sentences,total_sent_len):
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
        idf_score.update((x, math.log(int(total_sent_len) / y+1)) for x, y in idf_score.items())
        return idf_score
    def calculate_tf_IDF_score(self, tf_score, idf_score):
        tf_idf_score = {key: tf_score[key] * idf_score.get(key, 0) for key in tf_score.keys()}
        return tf_idf_score

    def Stem_rules_generation(self, str):
        str=str.strip(string.punctuation)
        str=str.replace(',', '')
        str = str.replace('.', '')
        str = str.replace(';', '')
        rubirc = []
        # print("qualified answer",doc)
        data_i_j = str.split(";")
        # print(data_i_j)
        # rules=[]
        dict1 = {}
        for j in range(len(data_i_j)):
            total_words = data_i_j[j].split()
            total_word_length = len(total_words)
            total_sentences = tokenize.sent_tokenize("".join(data_i_j[j]))
            total_sent_len = len(total_sentences)
            tf_score = self.caculate_tf_score(total_words, stop_words, total_word_length)
            idf_score = self.caculate_IDF_score(total_words, stop_words, total_sentences, total_sent_len)
            tf_idf_score = self.calculate_tf_IDF_score(tf_score, idf_score)
            keywords = self.get_top_n(tf_idf_score, total_word_length)
            # rules.append(keywords)
            dict1.update(keywords)
            # print("keywords_TF_IDF", keywords.values())
        sum1 = sum(dict1.values())
        for key in dict1:
            dict1[key] = dict1[key] / sum1
        rubirc.append(dict1)
        list_rules=list(dict1.keys())
        #print(list_rules)
        rules = {"rules":list_rules }
        return rubirc, rules


if __name__ == '__main__':
    #str="Let f(0)=0 and f(1)=1 then f(n)=f(n-1)+f(n-2)."
    #str="According to the condition, a/b+b/a=2; so a^2+b^2=2ab;a^2+ab+b^2=3ab, and a^2+4ab+b^2=6ab; therefore (a^2+ab+b^2)/(a^2+4ab+b^2)=1/2"
    #str="According to the conditions, we can get a+b=1, a-b=1 or a-b=-1; if a+b=1 and a-b=1, a=1, b=0, a^2019+b^2019=1. Otherwise, a+b=1, a-b=-1, a=0, b=1, a^2019+b^2019=1."
    str="(x+1)^2=9, we can obtain x+1=3 or x+1=-3, so x=2 or x=-4."
    math_r=Math_rubric()
    rubric1,rules = math_r.Stem_rules_generation(str)
    print(rules)
    #print(rubric1)
