from config import AppConfig
from app import HandControlApp


def main():
    app = HandControlApp(AppConfig())
    app.run()


if __name__ == "__main__":
    main()