from flask import Flask, request, render_template
import tensorflow as tf
import pickle


app = Flask(__name__)

@app.route('/')
def choose_prediction_method():
    return render_template('main.html')

def upr_prediction(params):
    # model = tf.keras.models.load_model('models/net_upr')
    model = pickle.load(open('models/best_model_upr.pkl', 'rb'))
    pred = model.predict([params])
    return pred

def pr_prediction(params):
    # model = tf.keras.models.load_model('models/net_pr')
    model = pickle.load(open('models/best_model_pr.pkl', 'rb'))
    pred = model.predict([params])
    return pred

def mn_prediction(params):
    # model = tf.keras.models.load_model('models/net_mn')
    model = pickle.load(open('models/best_model_mn.pkl', 'rb'))
    pred = model.predict([params])
    return pred

@app.route('/upr/', methods=['POST', 'GET'])
def upr_predict():
    message = ''
    if request.method == 'POST':
        param_list = ('mn', 'plot', 'mup', 'ko', 'seg', 'tv', 'pp', 'pr', 'ps', 'shn', 'pln')
        params = []
        for i in param_list:
            param = request.form.get(i)
            params.append(param)
        params = [float(i.replace(',', '.')) for i in params]

        message = f'Спрогнозированное значение модуля упругости при растяжении для введённых параметров: {upr_prediction(params)} ГПа'
    return render_template('upr.html', message=message)

@app.route('/pr/', methods=['POST', 'GET'])
def pr_predict():
    message = ''
    if request.method == 'POST':
        param_list = ('mn', 'plot', 'mup', 'ko', 'seg', 'tv', 'pp', 'mup', 'ps', 'shn', 'pln')
        params = []
        for i in param_list:
            param = request.form.get(i)
            params.append(param)
        params = [float(i.replace(',', '.')) for i in params]

        message = f'Спрогнозированное значение прочности при растяжении для введённых параметров: {pr_prediction(params)} МПа'
    return render_template('pr.html', message=message)

@app.route('/mn/', methods=['POST', 'GET'])
def mn_predict():
    message = ''
    if request.method == 'POST':
        param_list = ('plot', 'mup', 'ko', 'seg', 'tv', 'pp', 'mup', 'pr', 'ps', 'shn', 'pln')
        params = []
        for i in param_list:
            param = request.form.get(i)
            params.append(param)
        params = [float(i.replace(',', '.')) for i in params]

        message = f'Спрогнозированное cоотношение матрица-наполнитель для введённых параметров: {mn_prediction(params)}'
    return render_template('mn.html', message=message)

if __name__ == '__main__':
    # app.debug = True
    app.run()
