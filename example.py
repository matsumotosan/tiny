from tinygrad.engine import Value
from tinygrad.draw_graph import draw_graph


def main():
    a = Value(1.0) 
    b = Value(2.0) 
    c = Value(3.0) 
    d = Value(4.0) 

    print(a)
    print(b)
    print(c)
    print(d)

    ab = a + b
    cd = c + d

    print(ab)
    print(cd)

    abcd = ab * cd

    print(abcd)

    abcd.backward()
    draw_graph(abcd)


if __name__ == "__main__":
    main()
