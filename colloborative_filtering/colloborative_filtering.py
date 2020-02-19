import pandas as pd
metadata = pd.read_csv('D://TasteGraph/selfproject/movies_metadata.csv',low_memory =False)
#print the columns in the dataframe
#print (list(metadata))
#print (metadata['overview']).head()

#library to vectorize
from sklearn.feature_extraction.text import TfidfVectorizer

test_data=metadata.head(20)

tfidf =TfidfVectorizer(stop_words='english')
test_data['overview'] = test_data['overview'].fillna('')

tf_idf_matrix=tfidf.fit_transform(test_data['overview'])


#calculate the similarity scores
from sklearn.metrics.pairwise import linear_kernel

#compute the cosine similarity matrix
cosine_sim = linear_kernel(tf_idf_matrix,tf_idf_matrix)

print (test_data['overview'])
print (tf_idf_matrix)
print (cosine_sim)