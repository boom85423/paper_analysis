import requests
from bs4 import BeautifulSoup
import jieba.analyse
import multiprocessing as mp
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from gensim import corpora, models, similarities

def get_pages(home):
    # Get all pages about the home.
    req = requests.get(home, headers=headers)
    req.encoding = 'utf-8'
    soup = BeautifulSoup(req.text, 'lxml')
    res = soup.find_all('a', {'class':'step'})
    dom = 'https://researchoutput.ncku.edu.tw'
    pages = [home]
    for i in res:
        pages.append(dom + i['href'])
    return pages

def get_links(url):
    # Get all links in page.
    req = requests.get(url, headers=headers)
    req.encoding = 'utf-8'
    soup = BeautifulSoup(req.text, 'lxml')
    res = soup.find_all('h2', {'class':'title'})
    links = []
    for i in range(len(res)):
        links.append(res[i].select('a')[0]['href'])
    return links

def get_abstract(links):
    # Get abstract about the paper if existed.
    req = requests.get(links, headers=headers)
    req.encoding = 'utf-8'
    soup = BeautifulSoup(req.text, 'lxml')
    try:
        abstract = soup.select('.textblock p')[0].text
        return abstract
    except: # no abstract
        return '' # blank for join

def get_focus(links):
    # Get top5 TF-IDF about the paper if existed.
    req = requests.get(links, headers=headers)
    req.encoding = 'utf-8'
    soup = BeautifulSoup(req.text, 'lxml')
    abstracts = soup.select('.textblock p')
    try:
        res = jieba.analyse.extract_tags(abstracts[0].text, topK=5, withWeight=True)
        return res
    except: # no abstract
        return None

def get_similar_paper(target, number=3):
    # Use TF-IDF to recommend the user to read another paper.
    print('crawling...')
    doc_test = get_abstract(target)        
    test_doc_list = []
    for i in doc_test.split(' '):
        test_doc_list.append(i)
    
    all_doc_list = []
    for i in all_doc:
        doc_list = []
        for j in i.split(' '):
            doc_list.append(j)
        all_doc_list.append(doc_list)

    print('analyzing...')
    mydict = corpora.Dictionary(all_doc_list)
    corpus = []
    for i in all_doc_list:
        corpus.append(mydict.doc2bow(i))
    test_vec = mydict.doc2bow(test_doc_list)
    tfidf = models.TfidfModel(corpus)
    index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=len(mydict.keys()))
    sim = index[tfidf[test_vec]]   

    res = []
    for i in sorted(enumerate(sim), key = lambda item : item[-1], reverse=True):
        res.append(i[0])
    similar_links = []
    for i in res[0:number]:
        similar_links.append(links_dict[i])        
    return similar_links

if __name__ == '__main__':
    home = 'https://researchoutput.ncku.edu.tw/zh/persons/cheng-te-li/publications/'
    headers = {'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.99 Safari/537.36'}
    pages = get_pages(home)
    links = []
    for i in pages:
        links += get_links(i)
    
    #### word cloud for the professor ####
    # pool = mp.Pool()
    # abstract_jpb = [pool.apply_async(get_abstract, args=(i,)) for i in links]
    # abstract = [i.get() for i in abstract_jpb]
    # res = ' '.join(abstract)
    # cloud = WordCloud(background_color='white')
    # cloud.generate(res)
    # plt.imshow(cloud)
    # plt.axis('off')
    # plt.savefig('/Users/kevin102575/Desktop/wordcloud.png')
    # plt.show()

    #### TF-IDF ####
    # pool = mp.Pool()
    # focus_job = [pool.apply_async(get_focus, args=(i,)) for i in links]
    # focus = [i.get() for i in focus_job]
    # for i,j in zip(range(len(focus)), focus):
    #     print('------paper {}------\n'.format(i), j)

    #### Find similar papers ####
    pool = mp.Pool()
    target = 'https://researchoutput.ncku.edu.tw/zh/publications/analyzing-social-event-participants-for-a-single-organizer' # interested paper
    links.remove(target)
    links_dict = {}
    for i in enumerate(links):
        links_dict[i[0]] = i[1]
    abstract_jpb = [pool.apply_async(get_abstract, args=(i,)) for i in links]
    all_doc = [i.get() for i in abstract_jpb]
    similar_links = get_similar_paper(target)
    for i in similar_links:
        print(i)
