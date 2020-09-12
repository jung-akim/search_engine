import flask
from flask import request
from predictor_api import *
from collections import defaultdict

# Initialize the app
app = flask.Flask(__name__)

SHOW_K_REVIEWS = 5

data_dict = defaultdict()

# An example of routing:
# If they go to the page "/" (this means a GET request
# to the page http://127.0.0.1:5000/), return a simple
# page that says the site is up!
@app.route("/")
def hello():
    return "It's alive!!!"


@app.route("/predict", methods=["POST", "GET"])
def predict():

    input = request.json
    if input:
        global query, emphasized_keywords, index, res, data_dict, first# Flag for initializing data_dict only for the first time the query was entered.
        index = int(input["index"])
        first = bool(int(input['first']))
        if index == 0 and first:
            query, emphasized_keywords = input["query"], input["emphasized_keywords"]
            if query:
                if len(emphasized_keywords) > 0:
                    emphasized_keywords = emphasized_keywords.split(',')
                print(query)
                print(emphasized_keywords)
                show_k_reviews = SHOW_K_REVIEWS# How many (pros, cons) pairs of reviews to show?
                res = Results(emphasized_keywords = emphasized_keywords, show_k_reviews = show_k_reviews)
                res.match_products(query)
                global num_products
                num_products = len(res.matching_products)
                print(f"\nThere are {num_products} matched products from this query.")
                data_dict = defaultdict()

        if index >= 0:
            if index not in data_dict.keys():
                matched_product_title, k_pros, k_cons = res.get_reviews(index)
                matched_product_title = \
                    '<p id="product_block">'+\
                    '<span id="before"> < </span>'+\
                    '<span id="product_title"><strong>'+str(index + 1)+".</strong>"+matched_product_title+'</span>'+\
                    '<span id="after"> > </span>'+\
                    '</p>'

                k_pros = k_pros.style.render()
                k_cons = k_cons.style.render()

                num_products = '<span id = "num_products" style = "display:none;">' + str(num_products) + '</span>'

                data_dict[index] = matched_product_title, k_pros, k_cons, num_products

            matched_product_title, k_pros, k_cons, num_products = data_dict[index]

            return flask.render_template('predictor.html',
                                         matched_product_title = matched_product_title,
                                         k_pros = k_pros,
                                         k_cons = k_cons,
                                         num_products = num_products)
    return flask.render_template('predictor.html')
# Start the server, continuously listen to requests.
# We'll have a running web app!

# For local development:
if __name__ == '__main__':
    app.run(debug=True)

    # For public web serving:
    # app.run(host='0.0.0.0')
