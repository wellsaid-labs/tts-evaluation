from src.www.app import app
from src.www.app import cache_models

if __name__ == "__main__":
    cache_models()
    app.run()
