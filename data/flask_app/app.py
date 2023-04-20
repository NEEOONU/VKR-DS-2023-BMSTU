from flask import Flask, request, render_template
import tensorflow as tf
import pickle
app = Flask(__name__)

@app.route('/')
def choose_prediction_method():
    return render_template('main.html')

def upr_prediction(params):
    #загружаем нормализатор входных значений (вводимых параметров)
    scaler_in_upr = pickle.load(open('models/scaler_in_upr.pkl', 'rb'))
    #загружаем модель расчёта
    # model = tf.keras.models.load_model('models/net_upr')
    model = pickle.load(open('models/best_model_upr.pkl', 'rb'))
    #загружаем первую часть денормализатора
    scaler_out_upr1 = pickle.load(open('models/scaler_out_upr1.pkl', 'rb'))
    #загружаем вторую часть денормализатора
    scaler_out_upr2 = pickle.load(open('models/scaler_out_upr2.pkl', 'rb'))
    pred = model.predict(scaler_in_upr.transform([params]))
    #выдаём предсказание денормализованного вида, то есть не трансформированное значение предсказания с исходным масштабом
    pred_out = pred * scaler_out_upr1 + scaler_out_upr2
    return pred_out

def pr_prediction(params):
    #загружаем нормализатор входных значений (вводимых параметров)
    scaler_in_pr = pickle.load(open('models/scaler_in_pr.pkl', 'rb'))
    #загружаем модель расчёта
    # model = tf.keras.models.load_model('models/net_pr')
    model = pickle.load(open('models/best_model_pr.pkl', 'rb'))
    #загружаем первую часть денормализатора
    scaler_out_pr1 = pickle.load(open('models/scaler_out_pr1.pkl', 'rb'))
    #загружаем вторую часть денормализатора
    scaler_out_pr2 = pickle.load(open('models/scaler_out_pr2.pkl', 'rb'))
    pred = model.predict(scaler_in_pr.transform([params]))
    #выдаём предсказание денормализованного вида, то есть не трансформированное значение предсказания
    pred_out = pred * scaler_out_pr1 + scaler_out_pr2
    return pred_out

def mn_prediction(params):
    #загружаем нормализатор входных значений (вводимых параметров)
    scaler_in_mn = pickle.load(open('models/scaler_in_mn.pkl', 'rb'))
    #загружаем модель расчёта
    # model = tf.keras.models.load_model('models/net_mn')
    model = pickle.load(open('models/best_model_mn.pkl', 'rb'))
    #загружаем первую часть денормализатора
    scaler_out_mn1 = pickle.load(open('models/scaler_out_mn1.pkl', 'rb'))
    #загружаем вторую часть денормализатора
    scaler_out_mn2 = pickle.load(open('models/scaler_out_mn2.pkl', 'rb'))
    pred = model.predict(scaler_in_mn.transform([params]))
    #выдаём предсказание денормализованного вида, то есть не трансформированное значение предсказания
    pred_out = pred * scaler_out_mn1 + scaler_out_mn2
    return pred_out


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
        param_list = ('mn', 'plot', 'mup', 'ko', 'seg', 'tv', 'pp', 'mupr', 'ps', 'shn', 'pln')
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
        param_list = ('plot', 'mup', 'ko', 'seg', 'tv', 'pp', 'mupr', 'pr', 'ps', 'shn', 'pln')
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
