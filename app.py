#
# Flask form을 통해서 입력을 받고 머신러닝 예측을 해본다.
#

# 설치 필요: pip install scikit-learn     # 주의: pickle 모델을 만들 때 사용된 버전과 매치되어야 한다.
# 설치 필요: pip install numpy

from flask import Flask, render_template, url_for, session, redirect
from flask_wtf import FlaskForm
from wtforms import FloatField, SubmitField
from wtforms.validators import DataRequired, NumberRange
import pickle
import numpy as np                          

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret'          # 보안 설정 (필수).

# y의 유형 label을 list로 저장해 둔다.
y_labels = ["setosa", "versicolor", "virginica"]

# Feature 이름.
features_x = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

# 데이터 전처리 객체 읽어오기.
with open("my_scaler.pkl","rb") as f:
    scaler = pickle.load(f)

# 머신러닝 객체 읽어오기.
with open("my_classifier.pkl","rb") as f:
    classifier = pickle.load(f)

class MyForm(FlaskForm):
    sepalLength = FloatField(label='SEPAL LENGTH (4.3~7.9) : ', default=6., validators=[DataRequired(),NumberRange(min=4.3, max=7.9)])     
    sepalWidth = FloatField(label='SEPAL WIDTH (2.0~4.4) : ', default=3., validators=[DataRequired(),NumberRange(min=2.0, max=4.4)])     
    petalLength = FloatField(label='PETAL LENGTH (1.0~6.9) : ', default=4., validators=[DataRequired(),NumberRange(min=1.0, max=6.9)])     
    petalWidth = FloatField(label='PETAL WIDTH (0.1~2.5) : ', default=1., validators=[DataRequired(),NumberRange(min=0.1, max=2.5)])      
    submit = SubmitField(label= '제출')

@app.route('/', methods=['GET', 'POST'])
def index():
    aForm = MyForm()
    
    if aForm.validate_on_submit():                      # Form 제출됨??
        sepal_length = aForm.sepalLength.data
        sepal_width = aForm.sepalWidth.data
        petal_length = aForm.petalLength.data
        petal_width = aForm.petalWidth.data  
        x_raw = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        x_scaled = scaler.transform(x_raw)                              # Min-Max scaling. fit_transform이 아닌 transform!!
        session['y_prediction'] = y_labels[classifier.predict(x_scaled)[0]]         # 예측된 y 레이블.
        return redirect(url_for('prediction'))                                      # 예측된 결과를 보여준다.

    return render_template('index.html', form = aForm)

@app.route('/prediction/')
def prediction():
    return render_template('prediction.html')

if __name__ == '__main__':
    app.run(debug=False)              # 주의!!!!