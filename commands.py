from db.requests import make_dataset_O, make_dataset_X
from m_bkp.model import NN_bkp

def det(command):
    if command == "make dataset O":
        make_dataset_O()
    elif command == "make dataset X":
        make_dataset_X()
    elif command == "start model bkp":
        NN_bkp()
