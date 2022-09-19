from time import time

def a():
    global T1
    T1 = time()

def b():
    try:
        dt = time() - T1
        print(f"\nTime elapsed: {dt:.1f} seconds\n")
    except:
        print(f"\nMissing timeit.a()")

start = a
end = b
