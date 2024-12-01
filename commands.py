from db.requests import make_dataset_O, make_dataset_X


def det(command):
    if command == "make_dataset_O":
        make_dataset_O()
    elif command == "make_dataset_X":
        make_dataset_X()
