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

    def get_cursor(self):
        return self._db.cursor()

    def get_db(self):
        return self._db

    def insert_user(self,chat_id, stage):
        self._my_cursor.execute("INSERT INTO User (chat_id, cur_stage) VALUES (%s, %s)", (chat_id, stage))
        self._db.commit()

    def delete_user(self, cursor, chat_id):
        self._my_cursor.execute("DELETE FROM User WHERE chat_id = %s", chat_id)
        self._db.commit()

    def get_stage(self, cursor, chat_id):
        self._my_cursor.execute("SELECT cur_stage FROM User WHERE chat_id = %s", chat_id)
        return cursor.fetchall()[0][0]  # for return a string and not a tuple

    def update_stage(self, chat_id, stage):
        self._my_cursor.execute("UPDATE User SET cur_stage = %s WHERE chat_id = %s", (stage, chat_id))
        self._db.commit()



"test"
db = ProjectDB()
my_cursor = db.get_cursor()
# db.insert_user( 123, "hello")
db.update_stage(123, "goodbye")
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