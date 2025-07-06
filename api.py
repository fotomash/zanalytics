
from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'service': 'ZANFLOW v12 API'})

@app.route('/api/status')
def status():
    return jsonify({
        'status': 'running',
        'version': 'v12',
        'services': {
            'api': 'active',
            'dashboard': 'pending'
        }
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
