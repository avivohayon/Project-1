def beauty(func):
    def inner(*args):
        print("I DONT KNOW WHY")
        print(func(*args))
    return inner

@beauty
def double(x):
    return x*2

@beauty
def doda(what_is_aviv="GAE"):
    return f"Aviv is {what_is_aviv}"

if __name__ == '__main__':
    doda()