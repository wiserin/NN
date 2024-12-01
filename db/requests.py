from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from db.models import Dataset, X, O
import json

engine = create_engine('sqlite:///bot.db')

session = sessionmaker(bind=engine)
s = session()


# сериализация и десериализация списков
def serialize_list(lst):
    return json.dumps(lst)


def deserialize_list(s):
    return json.loads(s)


def make_dataset_X():
    for i in s.query(Dataset.field,
                     Dataset.move).filter(Dataset.player == 'X',
                                          Dataset.winner == 'X'):

        field = deserialize_list(i[0])
        move = deserialize_list(i[1])

        field[move[0]][move[1]] = None

        for j in range(0, 5):
            for k in range(0, 5):
                cell = field[j][k]
                if cell is None:
                    field[j][k] = 0
                elif cell == 'X':
                    field[j][k] = 1
                elif cell == 'O':
                    field[j][k] = -1

        new = X(field=serialize_list(field),
                move=serialize_list(move))
        s.add(new)
        s.commit()
    print('success')


def make_dataset_O():
    for i in s.query(Dataset.field,
                     Dataset.move).filter(Dataset.player == 'O',
                                          Dataset.winner == 'O'):

        field = deserialize_list(i[0])
        move = deserialize_list(i[1])

        field[move[0]][move[1]] = None

        for j in range(0, 5):
            for k in range(0, 5):
                cell = field[j][k]
                if cell is None:
                    field[j][k] = 0
                elif cell == 'X':
                    field[j][k] = -1
                elif cell == 'O':
                    field[j][k] = 1

        new = O(field=serialize_list(field),
                move=serialize_list(move))
        s.add(new)
        s.commit()
    print('success')
