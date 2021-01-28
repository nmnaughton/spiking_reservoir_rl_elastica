import multiprocessing as mp


def worker_func(in_q, out_q):
    while True:
        print(in_q.get())
        out_q.put(1)

if __name__ == "__main__":
    out_q = mp.Queue()
    in_q = mp.Queue()

    p = mp.Process(target=worker_func, args=(in_q, out_q))
    p.start()

    while True:
        in_q.put(0)
        print(out_q.get())
