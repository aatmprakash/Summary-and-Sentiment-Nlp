from flask import Flask, render_template, request
from senti2 import mainLoad

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("ch.html")


@app.route("/predict", methods=["POST"])
def predict():
    url = request.form.get('content')
    summary, cnt, rate =  mainLoad(url)
    return render_template("ch.html", summ_prediction=summary,cnt_prediction=cnt,rate_prediction=rate, url=url)

if __name__ == '__main__':
    app.run(host='0.0.0.0',port =80, debug=True)


#  pip install -r requirements.txt
