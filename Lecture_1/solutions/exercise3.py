#!/usr/bin/env python

def main():
    a = 1
    b = 1
    target = 48
    for n in range(3, 49):
        fib = a + b
        print(f"F({n}) = {fib}")
        a = b
        b = fib


if __name__ == "__main__":
    main()
