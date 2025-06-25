#!/usr/bin/env python3
"""Minimal ncOS local engine for quick testing"""

from flask import Flask, jsonify

app = Flask(__name__)

@app.get('/status')
def status():
    """Health check endpoint"""
    return jsonify({
        'service': 'ncOS Local Engine',
        'status': 'online',
        'version': '1.0.0'
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
