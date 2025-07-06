#!/usr/bin/env python3
"""Quick test to verify all imports work"""

print("Testing all imports...")

try:
    import streamlit
    print("✅ streamlit")
except ImportError as e:
    print(f"❌ streamlit: {e}")

try:
    import flask
    print("✅ flask")
except ImportError as e:
    print(f"❌ flask: {e}")

try:
    from flask_cors import CORS
    print("✅ flask-cors")
except ImportError as e:
    print(f"❌ flask-cors: {e}")

try:
    import pandas
    print("✅ pandas")
except ImportError as e:
    print(f"❌ pandas: {e}")

try:
    import numpy
    print("✅ numpy")
except ImportError as e:
    print(f"❌ numpy: {e}")

try:
    import plotly
    print("✅ plotly")
except ImportError as e:
    print(f"❌ plotly: {e}")

try:
    import yaml
    print("✅ yaml (pyyaml)")
except ImportError as e:
    print(f"❌ yaml (pyyaml): {e}")

try:
    import requests
    print("✅ requests")
except ImportError as e:
    print(f"❌ requests: {e}")

try:
    import sklearn
    print("✅ sklearn (scikit-learn)")
except ImportError as e:
    print(f"❌ sklearn (scikit-learn): {e}")

try:
    import matplotlib
    print("✅ matplotlib")
except ImportError as e:
    print(f"❌ matplotlib: {e}")

try:
    import seaborn
    print("✅ seaborn")
except ImportError as e:
    print(f"❌ seaborn: {e}")

print("\nAll imports successful! You can run ZANFLOW now.")
