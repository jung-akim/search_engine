from collections import defaultdict
from app_utils import *

reviews = pd.read_pickle('data/products_10_or_more_reviews.pkl')
unique_products = reviews[['product_parent','product_title']].drop_duplicates()
TOP_K_PRODUCTS = 5
TOP_K_REVIEWS = 5
SHOW_K_REVIEWS = 5
MAXIMUM_LEN = 300 # Maximum length of a review that will not be summarized

class Results:

    def __init__(self, top_k_products = TOP_K_PRODUCTS, top_k_reviews = TOP_K_REVIEWS, emphasized_keywords = None, show_k_reviews = SHOW_K_REVIEWS):
        self.top_k_products = top_k_products # k most relevant products
        self.top_k_reviews = top_k_reviews # k most helpful reviews
        self.emphasized_keywords = emphasized_keywords # list of keywords that are more weighted in the query
        self.show_k_reviews = show_k_reviews # k most relevant pros and k most relevant cons about the product

    def match_products(self, query):
        """
        Within the categorically matched top_k_products,
        look at the top_k helpful reviews and save their "sentiment * similarity" scores.

        :param query:
        :return:
        """

        print(f"We will examine reviews of at most {self.top_k_products} products that are most similar to the query")

        matching_scores = defaultdict(pd.Series)
        unique_products = reviews[['product_parent', 'product_title']].drop_duplicates()
        most_similar_indices, sim_scores = get_similarity_score_with_product(query)
        matching_products = unique_products.iloc[most_similar_indices[:self.top_k_products], 0].values

        print("Processing the review of product...")
        for i, product_id in enumerate(matching_products):
            print(f"...{i + 1}", end='')
            reviews_list = \
            reviews.loc[reviews.product_parent == product_id, ['review_body', 'helpful_votes', 'review_date']]. \
                sort_values(ascending=False, by=['helpful_votes', 'review_date'])[:self.top_k_reviews]['review_body']
            matching_scores[product_id] = reviews_list.apply(sentimental_similarity_score_of_a_review,
                                                             args=(query, 'positive', self.emphasized_keywords))

        med_matching_scores = [np.nanmedian(v) for k, v in matching_scores.items()]

        self.query = query
        self.sim_scores = sim_scores
        self.matching_products = matching_products
        self.product_indices = np.argsort(med_matching_scores)[::-1]
        self.product_scores = np.sort(med_matching_scores)[::-1]

    def get_reviews(self, index = 0):

        matched_product_id = self.matching_products[self.product_indices[index]]
        product_title = reviews.loc[reviews.product_parent == matched_product_id, :].head(1).product_title.values[0]
        print(f"\nYour query matched with:\n{product_title}")
        print(
            f"\nThe median of the Geometric mean of Positive Similarity score between \"{self.query}\" and \"{product_title}\" is {self.product_scores[index]:.3f}")
        # Geometric mean takes into account the effect of compounding, therefore, better suited for calculating the returns.

        matched_reviews = reviews.loc[
            reviews.product_parent == matched_product_id, ['review_body', 'star_rating', 'review_date']]
        matched_reviews['review_date'] = matched_reviews['review_date'].apply(lambda x: str(x).split()[0])
        matched_reviews['star_rating'] = matched_reviews['star_rating'].apply(int)

        pos = (matched_reviews.star_rating == 5.0).astype(int)
        sim = matched_reviews.review_body.apply(lambda x: sentimental_similarity_score_of_a_review(x, self.query, None,
                                                                                                   self.emphasized_keywords) if x is not None else None)
        pos_sim = pd.concat([pos.rename('positive'), sim.rename('similarity')], axis=1)
        matched_reviews[['positive', 'similarity']] = pos_sim

        pos_sim = pd.concat([pos.rename('positive'), sim.rename('similarity')], axis=1)
        matched_reviews[['positive', 'similarity']] = pos_sim

        show_k_reviews = self.show_k_reviews

        pos_sim_indices = pos_sim.loc[pos_sim.positive == 1, 'similarity'].sort_values(ascending=False).head(show_k_reviews).index
        neg_sim_indices = pos_sim.loc[pos_sim.positive == 0, 'similarity'].sort_values(ascending=False).head(show_k_reviews).index

        k_pros = matched_reviews.loc[pos_sim_indices, ['review_body', 'star_rating', 'review_date', 'similarity']]
        k_cons = matched_reviews.loc[neg_sim_indices, ['review_body', 'star_rating', 'review_date', 'similarity']]

        mask = k_pros.review_body.map(lambda x: len(x)) >= MAXIMUM_LEN
        k_pros.loc[mask, 'original'] = k_pros.loc[mask, 'review_body'].values
        k_pros.loc[mask, 'review_body'] = k_pros.loc[mask, 'review_body'].apply(create_summary)
        k_pros.loc[:, 'similarity'] = k_pros.loc[:, 'similarity'].apply(lambda x: str(round(x, 2)) if not pd.isnull(x) else '')

        mask = k_cons.review_body.map(lambda x: len(x)) >= MAXIMUM_LEN
        k_cons.loc[mask, 'original'] = k_cons.loc[mask, 'review_body'].values
        k_cons.loc[mask, 'review_body'] = k_cons.loc[mask, 'review_body'].apply(create_summary)
        k_cons.loc[:, 'similarity'] = k_cons.loc[:, 'similarity'].apply(lambda x: str(round(x, 2)) if not pd.isnull(x) else '')

        return product_title, \
               k_pros.rename(columns = {'review_body':'Pros', 'star_rating':'Stars', 'review_date': 'Date', 'similarity': 'Similarity'}), \
               k_cons.rename(columns = {'review_body':'Cons', 'star_rating':'Stars', 'review_date': 'Date', 'similarity': 'Similarity'})