import datetime
import random

class MessageManger():
    def __init__(self):
        self.save_msg("----- Starting new session -----")

    def save_msg(self, msg):
        """
        saving the sent msg with the date
        :param msg: msg to save
        """
        with open("messages_log.txt", "a") as f:
            f.write(f"{datetime.datetime.now()}{msg}\n")

    def get_metal(self):
        with open("metal.txt") as f:
            lines = f.read().split()
        return lines[random.randint(0, len(lines)-1)]

    def set_metal(self, msg):
        with open("metal.txt", "a") as f:
            f.write(f"\n{msg}")
