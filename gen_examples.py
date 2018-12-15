import random

def gen_int(f):
    for j in range(0, random.randint(1, 10)):
            f.write(str(random.randint(1, 9)))

def gen_char(f, char):
    for j in range(0, random.randint(1, 10)):
            f.write(char)

def gen_pos(num):
    f = open("pos_examples", "w+")
    for i in range(num):
        gen_int(f)

        gen_char(f, "a")

        gen_int(f)

        gen_char(f, "b")

        gen_int(f)

        gen_char(f, "c")

        gen_int(f)

        gen_char(f, "d")

        gen_int(f)

        f.write('\n')

    f.close()


def gen_neg(num):
    f = open("neg_examples", "w+")
    for i in range(num):
        gen_int(f)

        gen_char(f, "a")

        gen_int(f)

        gen_char(f, "c")

        gen_int(f)

        gen_char(f, "b")

        gen_int(f)

        gen_char(f, "d")

        gen_int(f)

        f.write('\n')

    f.close()
# language starts and ends with the same 2 digits
def gen_pos_same_2digits(num):
    f=open("pos_same2_examples", "w+")
    for i in range(num):
        num = random.randint(0, 9)
        s = str(num) + str(num)
        length=random.randint(0,30)
        for j in range(length):
            s+=str(random.randint(0,9))
        s += str(num) + str(num)
        f.write(s+"\n")

    f.close()
def gen_neg_same_2digits(num):
    f = open("neg_same2_examples", "w+")
    for i in range(num):
        num=random.randint(0,9)
        num2=random.randint(0,9)
        while num==num2:
            num2=random.randint(0,9)
        s = str(num) + str(num2)
        length = random.randint(0, 32)
        for j in range(length):
            s += str(random.randint(0, 9))
        f.write(s + "\n")
    f.close()

# language starts and ends with the same 2 digits
def gen_pos_palindrom(num):
    f=open("pos_palindrom", "w+")
    for i in range(num):
        length = random.randint(1, 15)
        sen=''
        for j in range(length):
            sen+=str(random.randint(0,9))
        for j in range(length-1,-1,-1):
            sen+=sen[j]
        f.write(sen+"\n")
    f.close()

def gen_neg_palindrom(num):
    f = open("neg_palindrom", "w+")
    for i in range(num):
        length=random.randint(1,30)
        sen=''
        for j in range(length):
            sen+= str(random.randint(0,9))
        if sen[0]==sen[length-1]:
            sen+="1324"
        f.write(sen + "\n")
    f.close()



def main():
    # gen_pos(1000)
    # gen_neg(1000)
    # gen_pos_same_2digits(1000)
    # gen_neg_same_2digits(1000)
    gen_pos_palindrom(1000)
    gen_neg_palindrom(1000)
if __name__ == "__main__":
    main()