import collections


with open('98-0.txt', encoding="utf-8") as f:
    words_tobe_counted = f.read()

with open('stopwords') as f:
    stopwords = f.read()
word_count = {}
for word in words_tobe_counted.lower().split():
    word = word.replace(",","")
    word = word.replace("“","")
    word = word.replace(".","")
    if word not in stopwords:
        if word not in word_count:
            word_count[word] = 1
        else:
            word_count[word] +=1
d = collections.Counter(word_count)
print(d.most_common(3))
for word, count in d.most_common(10):
    print(word, ":", count)