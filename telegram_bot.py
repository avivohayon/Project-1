from flask import Flask, request, jsonify
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
import telebot
from telebot import types
from constant import TOKEN_API
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
import pdf_downlad
import project_db
import finance_analyzer
import os
bot = telebot.TeleBot(TOKEN_API, parse_mode=None)
states_dict ={"Menu": 0, "Start": 1, "All": 2, "Some": 3, "ML": 4, "Close": 5}
bot_db = project_db.ProjectDB()
crawler = pdf_downlad.Crawler()
crawler.init()
crawler.delete_prev_pdfs(pdf_downlad.output_dir_all)
crawler.delete_prev_pdfs(pdf_downlad.output_dir_some)

#TODO
# read on "docker"( la misaviv)


# use "@" for any function that our bot will use as msg handling
@bot.message_handler(commands=["hello", "start"])
def main(msg):
    bot_db.delete_user([msg.chat.id])
    register_user([msg.chat.id], "Start")
    bot.reply_to(msg, "Hello! Im Mr Buttons your furry finance bot!\n")
    markup = types.ReplyKeyboardMarkup(row_width=2, one_time_keyboard=True, resize_keyboard=True)
    btn1 = types.KeyboardButton("/all")
    btn2 = types.KeyboardButton("/some")
    btn3 = types.KeyboardButton("/close")
    markup.add(btn1, btn2, btn3)
    bot.send_message(chat_id=msg.chat.id, text=" What will you like me to do?", reply_markup=markup)


@bot.message_handler(commands=["all"])
def send_all_pdfs(msg):
    bot_db.update_state([msg.chat.id], "All")
    bot.reply_to(msg, "Getting all the file, Meow. it might take a second. here a song while you wait Meoww")
    bot.send_message(chat_id=msg.chat.id, text="https://www.youtube.com/watch?v=jIQ6UV2onyI")
    crawler.get_all()
    all_files = os.listdir(pdf_downlad.output_dir_all)
    cwd = os.getcwd() + pdf_downlad.output_dir_all + "\\"
    for file in all_files:
        cur_pdf = open(cwd+file, 'rb')
        bot.send_document(msg.chat.id, cur_pdf)
        cur_pdf.close()
#     #TODO
#     # i need to see if i need to delete all the file if someone will ask to get all, and then get just some so
#     # i wont give the user a file he doesnt need

#TODO write a decorator that saves user current menu state by chat_id

@bot.message_handler(commands=["some"])
def send_some_pdfs(msg):
    bot_db.update_state([msg.chat.id], "Some")
    bot.reply_to(msg, "Please name the companies you want separate by comma i.e: \nfor one company write  (with comma): comapny_name, \n "
                      "for more then one company write: company_name1,company_name2,... ")
    print(msg.text)

@bot.message_handler(commands=["close"])
def close_menu(msg):
    bot_db.update_state([msg.chat.id], "Close")
    markup = types.ReplyKeyboardRemove(selective=False)
    bot.send_message(chat_id=msg.chat.id, text="Meow Meow", reply_markup=markup )
    bot_db.delete_user([msg.chat.id])


@bot.message_handler()
def router(msg):
    cur_user_state = bot_db.get_state([msg.chat.id])
    # TODO user_menu_state (from db by chat_id)
    user_menu_state = states_dict[cur_user_state]
    match user_menu_state:
        case 0:
            # TODO call Main menu
            pass
        case 1:
            # TODO call start
            pass
        case 2:
            pass
        case 4:
            # TODO call ML
            pass
        case 3:
            get_companies_names(msg)
        case 5:
            close_menu(msg)



# @bot.message_handler(func=lambda msg:msg.text is not None and "," in msg.text)
def get_companies_names(msg):
    bot.reply_to(msg, "Working on it!")
    companies_lst = msg.text.split(",")
    print(companies_lst)
    crawler.get_some(companies_lst)
    all_files = os.listdir(pdf_downlad.output_dir_some)
    print(f"all files are: {all_files}")
    cwd = os.getcwd() + pdf_downlad.output_dir_some + "\\"
    for file in all_files:
        cur_pdf = open(cwd + file, 'rb')
        bot.send_document(msg.chat.id, cur_pdf)
        cur_pdf.close()




def register_user(chat_id, command):
    if bot_db.insert_user(chat_id, command):
        return True
    return False

# let the bot listen to msgs
bot.polling()



#TODO
# after execute some or all cmd need to return to give again the KeyboardButton to chose to quit or some