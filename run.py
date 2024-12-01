from db.models import start_db
from commands import det


def main():
    start_db
    while True:
        det(input())


if __name__ == "__main__":
    main()
