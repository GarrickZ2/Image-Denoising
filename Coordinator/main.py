from flask import Flask, request
import pickle
from coordinator import Coordinator
import base64
import numpy as np


app = Flask(__name__)
machine_num = 2

coordinator = Coordinator()


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route("/generator", methods=['post'])
def get_generator():
    data = request.form.get('loss')
    data = base64.b64decode(data)
    data = pickle.loads(data)
    state_dict = coordinator.update_generator(data)
    response_data = pickle.dumps(state_dict)
    return response_data


@app.route("/discriminator", methods=['post'])
def get_discriminator():
    data = request.form.get('loss')
    data = base64.b64decode(data)
    data = pickle.loads(data)
    state_dict = coordinator.update_discriminator(data)
    response_data = pickle.dumps(state_dict)
    return response_data


@app.route("/update", methods=['get'])
def get_update_scheduler():
    coordinator.call_scheduler()
    return 200


@app.route("/save", methods=['get'])
def save_result():
    path = request.args.get("path")
    coordinator.save(path)
    return 200


@app.route("/load", methods=['get'])
def load_result():
    path = request.args.get("path")
    ts = request.args.get("ts")
    coordinator.load(path, ts)
    return 200


@app.route("/validate", methods=['get'])
def update_validation():
    data = request.args.get('val')
    data = np.float32(data)
    coordinator.update_val(data)
    return 200


@app.route("/validate/finished", methods=['get'])
def finish_validation():
    return coordinator.finish_val()


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=6000)

