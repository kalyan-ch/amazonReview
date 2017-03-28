import pandas as pd
import re
from nltk.corpus import stopwords
from bs4 import BeautifulSoup as bs

def get_review_words(rev):

	rev_temp = bs(rev)
	rev_text = re.sub("[^a-zA-Z]"," ",rev_temp.get_text())
	rev_words = [w for w in rev_text.split() if not w in st_words]
	str1 = " ".join(rev_words)

	return str1
	

if __name__ == '__main__':
	
	#reading dataset
	data_raw = pd.read_csv("amazon_baby_train.csv")
	print(data_raw.shape)
	reviews = data_raw['review']
	rats = data_raw['rating']
	st_words = stopwords.words("english")

	reviews_ref = [] 
	for i in range(len(reviews)):
		rev = str(reviews[i])
		mean_review = get_review_words(rev)
		mean_review += " :raTe: "+str(rats[i])
		reviews_ref.append(mean_review)
	
	f = open("reviews_train.txt", "w")
	
	for x in reviews_ref:
		f.write(x+"\n")

	f.close()
	

