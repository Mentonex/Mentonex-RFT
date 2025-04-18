from flask import Flask, jsonify, request

from trinity.common.config import load_config
from trinity.data.controllers.active_iterator import DataActiveIterator

app = Flask(__name__)


@app.route("/data_workflow", methods=["GET"])
def data_workflow():
    config_path = request.args.get("configPath")
    config = load_config(config_path)

    iterator = DataActiveIterator(config)
    ret, msg = iterator.run()
    return jsonify({"return_code": ret, "message": msg})


if __name__ == "__main__":
    app.run(debug=True)
