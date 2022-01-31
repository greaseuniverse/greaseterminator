from flask import Flask, send_from_directory
import os

app = Flask(__name__)

@app.route('/<filename>')  
def send_file(filename):  
    print(os.getcwd())
    return send_from_directory('./img', filename)

if __name__ == '__main__':
    app.run(debug=True)