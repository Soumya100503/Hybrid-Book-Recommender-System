from flask import Flask,render_template,request
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from scipy.sparse.linalg import svds

popular_books_df = pickle.load(open('popular_books.pkl','rb'))
table = pickle.load(open('table.pkl','rb'))
books_df= pickle.load(open('books.pkl','rb'))
similarity_scores = pickle.load(open('similarity_scores.pkl','rb'))
content_df = pickle.load(open('content-based.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html',
                           book_name = list(popular_books_df['Book-Title'].values),
                           author=list(popular_books_df['Book-Author'].values),
                           image=list(popular_books_df['Image-URL-M'].values),
                           votes=list(popular_books_df['num_ratings'].values),
                           rating=list(popular_books_df['avg_ratings'].values)
                           )



@app.route('/recommend')
def recommend_ui():
    return render_template('recommend_books.html')

@app.route('/recommend_books',methods=['post'])
def recommend():
    book_title = request.form.get('user_input')
    def get_content_based_recommendations(book_title):
        tfidf_vectorizer = TfidfVectorizer()
        content_matrix = tfidf_vectorizer.fit_transform(content_df['Content'])


        svd = TruncatedSVD(n_components=100)  
        reduced_content_matrix = svd.fit_transform(content_matrix)

        try:
            index = content_df[content_df['Book-Title'] == book_title].index[0]
        except IndexError:
            return f'Book title "{book_title}" not found in the dataset.'

        similarity_scores = linear_kernel(reduced_content_matrix[index:index+1], reduced_content_matrix).flatten()
        similar_indices = similarity_scores.argsort()[::-1][1:21]

        valid_indices = [i for i in similar_indices if i < len(content_df)]
        
        recommendations = content_df.iloc[valid_indices]['Book-Title'].values
        return list(recommendations)
    
    def get_collaborative_based_recommendations(book_title):
        index = np.where(table.index == book_title)[0][0]
        similar_items = sorted(list(enumerate(similarity_scores[index])),key=lambda x:x[1],reverse=True)[1:21]

        recommended_movies = []
        for i in similar_items:
            item = []
            temp_df = books_df[books_df['Book-Title'] == table.index[i[0]]]
            item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
            item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
            item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))
            
            recommended_movies.append(item[0])
        return recommended_movies
    
    try:
        top_n = 10
        content_based_recommendations = get_content_based_recommendations(book_title)
        collaborative_filtering_recommendations = get_collaborative_based_recommendations(book_title)

        content_based_scores = {books: (top_n - i) * 0.3 for i, books in enumerate(content_based_recommendations)}
        collaborative_filtering_scores = {books: (top_n - i) * 0.7 for i, books in enumerate(collaborative_filtering_recommendations)}
        
        hybrid_scores = {}
        for books, score in content_based_scores.items():
            if books in hybrid_scores:
                hybrid_scores[books] += score
            else:
                hybrid_scores[books] = score
                
        for books, score in collaborative_filtering_scores.items():
            if books in hybrid_scores:
                hybrid_scores[books] += score
            else:
                hybrid_scores[books] = score
                
        sorted_hybrid_recommendations = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
        hybrid_recommendations = [books for books, score in sorted_hybrid_recommendations[:top_n]]
    except:
        hybrid_recommendations = get_content_based_recommendations(book_title)[:10]

    data = []
    for i in hybrid_recommendations:
        objects = []
        df = books_df[books_df['Book-Title'] == i]
        objects.extend(list(df.drop_duplicates('Book-Title')['Book-Title'].values))
        objects.extend(list(df.drop_duplicates('Book-Title')['Book-Author'].values))
        objects.extend(list(df.drop_duplicates('Book-Title')['Image-URL-M'].values))
 
        data.append(objects)



    print(data)

    return render_template('recommend_books.html',data=data)

if __name__ == '__main__':
    app.run(debug=True)

