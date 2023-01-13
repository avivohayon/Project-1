import mysql.connector


class ProjectDB:
    def __init__(self):
        self._db = mysql.connector.connect(
            host="localhost",
            user="root",
            passwd="dbproject1995",
            database="projectdb"
        )
        self._my_cursor = self._db.cursor()
        self._stage_set = {"Main_Menu", "All", "Some", "ML"}

    def get_cursor(self):
        return self._db.cursor()

    def get_db(self):
        return self._db

    def insert_user(self, chat_id, state):
        """
        insert user name to the database
        :param chat_id: telegram chat_id
        :param state: state of the user
        :return: if the chat_id (key) is already exists, return False, else add to DB and return True
        """
        if self.check_if_exist(chat_id):
            return False
        self._my_cursor.execute("INSERT INTO User (chat_id, cur_stage) VALUES (%s, %s)", (chat_id, state))
        self._db.commit()
        return True

    def delete_user(self, chat_id):
        self._my_cursor.execute("DELETE FROM User WHERE chat_id = %s", chat_id)
        self._db.commit()

    def get_state(self, chat_id):
        """
        return the user state (if exists)
        :param chat_id: user chat id
        :return: if the user exist return the state else return False
        """
        if self.check_if_exist(chat_id):
            self._my_cursor.execute("SELECT cur_stage FROM User WHERE chat_id = %s", chat_id)
            return self._my_cursor.fetchall()[0][0]
        return False
        # result = self._my_cursor.fetchall()
        # if result != [] : return result[0][0]
        # return False
        # return self._my_cursor.fetchall()[0][0]  # for return a string and not a tuple

    def update_state(self, chat_id, state):
        """
        update the state of a user (if exists)
        :param chat_id: the chat_id (key) of the user
        :param state: the state to update
        :return: True if the user exists and has been added, False otherwise
        """
        if self.check_if_exist(chat_id):
            self._my_cursor.execute("UPDATE User SET cur_stage = %s WHERE chat_id = %s", (state, chat_id))
            self._db.commit()
            return True
        return False
    def check_if_exist(self, chat_id):
        """
        checking if a user is already exists
        :param chat_id: telegram chat_id
        :return: True is exits, False otherwise
        """
        self._my_cursor.execute("SELECT * FROM User WHERE chat_id = %s", chat_id)
        if self._my_cursor.fetchall() != [] :return True
        return False



db = ProjectDB()
my_cursor = db.get_cursor()
# db.insert_user( 123, "hello")
# db.update_stage(123, "goodbye")
print(db.insert_user(123, "asd"))
# stage = db.get_stage(my_cursor, [123])
# print(stage[0][0])
# db.delete_user(my_cursor, [123])

# my_cursor.execute("SELECT * FROM User")


# my_cursor = db.cursor()
#
# # my_cursor.execute("CREATE TABLE User (chat_id INTEGER(11) UNSIGNED PRIMARY KEY , cur_stage VARCHAR (30))")
# my_cursor.execute("DESCRIBE User")
# # my_cursor.execute("CREATE DATABASE projectdb")
#
# # my_cursor.execute("SHOW DATABASES")
# for x in my_cursor:
#     print(x)
# add_formula = ""