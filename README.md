# wordpreprocessing_wordcloud
    import pandas as pd 
    import numpy as np
    import string 
    import re #regex library
    from nltk.tokenize import word_tokenize 
    from nltk.probability import FreqDist
    from nltk.tokenize import word_tokenize 
    from nltk.probability import FreqDist
    import nltk
    nltk.download('stopwords')
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    import swifter
    from nltk.corpus import stopwords

     komen = pd.DataFrame(pd.read_excel("C:/Galuh/komenig.xlsx")) #change path
     komen.head()

# ------ Case Folding --------
    komen['komen'] = komen['komen'].str.lower()


    print('Case Folding Result : \n')
    print(komen['komen'].head(5))
    print('\n\n\n')

# ------ Tokenizing ---------

    def remove_tweet_special(text):
    # remove tab, new line, ans back slice
    text = text.replace('\\t'," ").replace('\\n'," ").replace('\\u'," ").replace('\\',"")
    # remove non ASCII (emoticon, chinese word, .etc)
    text = text.encode('ascii', 'replace').decode('ascii')
    # remove mention, link, hashtag
    text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)"," ", text).split())
    # remove incomplete URL
    return text.replace("http://", " ").replace("https://", " ")
                
    komen['komen'] = komen['komen'].apply(remove_tweet_special)

#remove number
    def remove_number(text):
    return  re.sub(r"\d+", "", text)

    komen['komen'] = komen['komen'].apply(remove_number)

#remove punctuation
    def remove_punctuation(text):
     return text.translate(str.maketrans("","",string.punctuation))

    komen['komen'] = komen['komen'].apply(remove_punctuation)

#remove whitespace leading & trailing
    def remove_whitespace_LT(text):
     return text.strip()

    komen['komen'] = komen['komen'].apply(remove_whitespace_LT)

#remove multiple whitespace into single whitespace
    def remove_whitespace_multiple(text):
        return re.sub('\s+',' ',text)

    komen['komen'] = komen['komen'].apply(remove_whitespace_multiple)

# remove single char
    def remove_singl_char(text):
     return re.sub(r"\b[a-zA-Z]\b", "", text)

    komen['komen'] = komen['komen'].apply(remove_singl_char)

# NLTK word rokenize 
    def word_tokenize_wrapper(text):
        return word_tokenize(text)

    komen['komen_tokens'] = komen['komen'].apply(word_tokenize_wrapper)

    print('Tokenizing Result : \n') 
    print(komen['komen_tokens'].head())
    print('\n\n\n')

# ----------------------- get stopword from NLTK stopword -------------------------------
    # get stopword indonesia
    from nltk.corpus import stopwords
    list_stopwords = stopwords.words('indonesian')


# append additional stopword
    list_stopwords.extend(['assalamualaikum', 'hayu', 'mba', 'ala', 'pas', 'nanabisa', 'batalbrsekarang', 'rakyat', 'nana',
                       'andovi', 'dovi', 'jovial', 'jovi', 'najwa', 'karaoke', 'gorden', 'uang', 'ganti', 'corong', 'lho', 
                       'benerbener', 'yah', 'ka', 'kak', 'ya', 'yak', 'orgquot', 'nya', 'hayuu', 'si', 'mbak', 'mas', 'abang', 
                       'kang','bro', 'sis', 'sist', 'and', 'dan', 'of', 'the','it', 'is', 'my', 'you', 'kalibrbroh', 
                       'worldbelieveunder','mistletoepurposechangesjustice', 'ngewibu', 'its', 'thats', 'tbtb', 'jo',
                       'oreabralurnya', 'dp', 'militerbrsnowdrop', 'politikbrhometown', 'chachacha', 'kabs', 'whats', 'up', 
                       'guys', 'ourblues', 'in', 'cell','amp', 'nih', 'noh', 'itu', 'ini', 'sini', 'kan', 'benerquot', 'deh', 
                       'ko', 'cintabrbrsbnrnya', 'hehehe', 'wkwkbrapalagi', 'tk', 'vagabond', 'siapquot', 'haha', 
                       'assalamualikum', 'bralhamdulilah', 'oh', 'pribadibrbrmakasih','wkwkwkwk', 'from', 'kehidupanbrbrjadi',
                       'orderbrbr', 'siasiabrbr', 'upnya', 'gofoodbrbr', 'siapsiap', 'ah', 'oh', 'eh', 'ih',
                       'squid', 'romancebrbrselamat', 'shihabbrini', 'blablablaabrmasa', 'fo', 'bo', 'ak', 'siih', 
                       'ramahbbrbrmgetuk','yang','kalau', 'sih', 'to','yg', 'dr', 'dri', 'dari', 'brbr', 'on','br', 'jd', 'mb',
                      'najwashihab', 'andovidalopez', 'jovialdalopez', 'shihab', 'kl', 'klo', 'kli','lg', 'ri', 'mu', 'ken', 
                      'tidak', 'gak', 'ga', 'tdk', 'min', 'knp', 'pd', 'tuh', 'rm', 'kayak', 'rap', 'sj', 'saja', 'karena',
                      'karna', 'pke', 'kalau', 'dah', 'udah', 'kalau', 'sja','aja'])

     # read txt stopword using pandas
        txt_stopword = pd.read_csv("C:/Galuh/stopword.txt", names= ["stopword"], header = None)

     # convert stopword string to list & append additional stopword
         list_stopwords.extend(txt_stopword["stopword"][0].split(' '))

    # convert list to dictionary
        list_stopwords = set(list_stopwords)

    #remove stopword pada list token
     def stopwords_removal(words):
            return [word for word in words if word not in list_stopwords]

     komen['komen_token_sw'] = komen['komen_tokens'].apply(stopwords_removal) 

     print('Stopword Result: \n')
     print(komen['komen_token_sw'])
        print('\n\n\n')

# ------------------------------------normalization---------------------------------------------
    normalizad_word = pd.read_excel("C:/Galuh/normalisasi.xlsx") #change path

    normalizad_word_dict = {}

    for index, row in normalizad_word.iterrows():
     if row[0] not in normalizad_word_dict:
            normalizad_word_dict[row[0]] = row[1] 

    def normalized_term(document):
        return [normalizad_word_dict[term] if term in normalizad_word_dict else term for term in document]

    komen['komen_normalize']=komen['komen_token_sw'].apply(normalized_term)
    komen['komen_normalize'].head(10)

# -------------------------stemmer----------------------------
   #create stemmer
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

   #stemmed
    def stemmed_wrapper(term):
        return stemmer.stem(term)

    term_dict = {}
    for document in komen['komen_normalize']:
        for term in document:
            if term not in term_dict:
                term_dict[term] = ' '
            
    print(len(term_dict))
    print("------------------------")

    for term in term_dict:
        term_dict[term] = stemmed_wrapper(term)
        print(term,":" ,term_dict[term])
    
    print(term_dict)
    print("------------------------")


    #apply stemmed term to dataframe
    def get_stemmed_term(document):
     return [term_dict[term] for term in document]
    komen['komen_stemmed']=komen['komen_normalize'].swifter.apply(get_stemmed_term)
    print(komen['komen_stemmed'])

    komen.to_csv("C:/Galuh/prep.csv") #change path

# ====================WORD CLOUD==========================
    from wordcloud import WordCloud, STOPWORDS
    import matplotlib.pyplot as plt
    import pandas as pd

#Reads csv file
    df = pd.read_csv("C:/Galuh/prep.csv", encoding ="latin-1")
 
    comment_words = ''
    stopwords = set(STOPWORDS)

    #iterate through the csv file
    for val in df.komen_stemmed:
     
    # typecaste each val to string
    val = str(val)
 
    # split the value
    tokens = val.split()
     
    # Converts each token into lowercase
    for i in range(len(tokens)):
        tokens[i] = tokens[i].lower()
     
    comment_words += " ".join(tokens)+" "
    
    wordcloud = WordCloud(width = 800, height = 800,
                background_color ='white',
                stopwords = stopwords,
                min_font_size = 10).generate(comment_words)
    
    # plot the WordCloud image                      
    plt.figure(figsize = (8, 8), facecolor = None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad = 0)
 
    plt.show()
