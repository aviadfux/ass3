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

def main():
    gen_pos(500)
    gen_neg(500)

if __name__ == "__main__":
    main()