from db.requests import make_dataset_O, make_dataset_X
from m_bkp.model import NN_bkp
from m_ql.UI import GameUI, play_X, play_O
from m_ql.main import main
from m_ql.model_VS_model import main as main_1

def det(command):
    if command == "make dataset O":
        make_dataset_O()
    elif command == "make dataset X":
        make_dataset_X()
    elif command == "start model bkp":
        NN_bkp()
    elif command == "run UI":
        app = GameUI()
        app.run()
    elif command == "train DQL":
        main(1, 2)
        main(2, 1)
    elif command == "play X":
        play_X()
    elif command == "play O":
        play_O()
    elif command == "train main":
        main_1()

