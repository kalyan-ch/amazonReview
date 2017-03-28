
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import time

if __name__ == '__main__':
	with open("reviews_1.txt") as f:
		content = f.readlines()

	vectorizer = TfidfVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, max_features = 5000)
	
	clean_revs = []
	ratings = []
	for rev in content:
		index = rev.find(":raTe: ")
		ratings.append(int(str(rev[index+7:])))
		review = rev[:index]
		clean_revs.append(review)

	train_data_features = vectorizer.fit_transform(clean_revs)
	train_data_features = train_data_features.toarray()
	scaler = StandardScaler()
	scaler.fit(train_data_features)
	train_data_features = scaler.transform(train_data_features)

	start = time.clock()
	print("Training started")

	mlp = MLPClassifier(activation='logistic', learning_rate='adaptive', hidden_layer_sizes=(2500,2500), random_state=1)
	mlp.fit(train_data_features, ratings)

	print("Training complete!")
	print("Time taken: "+str((time.clock() - start)/60.0)+" minutes")
	
	with open("reviews_test.txt") as f:
		content = f.readlines()
	
	clean_revs = []
	ratings = []
	for rev in content:
		index = rev.find(":raTe: ")
		ratings.append(int(str(rev[index+7:])))
		review = rev[:index]
		clean_revs.append(review)

	test_data_features = vectorizer.fit_transform(clean_revs)
	test_data_features = test_data_features.toarray()
	scaler = StandardScaler()
	scaler.fit(test_data_features)
	test_data_features = scaler.transform(test_data_features)

	print("predition started")
	predicitons = mlp.predict(test_data_features)
	print("prediciton ended")
	count = 0
	for i in range(len(predicitons)):
		if(int(predicitons[i]) == ratings[i]):
			count += 1

	print(""+str(count/len(predicitons)*1.0))

	