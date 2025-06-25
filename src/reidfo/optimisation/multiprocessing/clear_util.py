import os


def clear_console():
    command = "cls" if os.name == "nt" else "clear"
    os.system(command)
