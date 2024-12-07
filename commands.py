from db.requests import make_dataset_O, make_dataset_X
from m_bkp.model import NN_bkp
from m_ql.UI import GameUI

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

