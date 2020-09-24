import os
from reservoir_rl import generate_cma_plots

def main():
    last_gen_dir =  os.path.join('./cma_es_data', os.listdir('./cma_es_data')[-1])
    generate_cma_plots(load_dir=last_gen_dir, save_dir='./BlueWatersResults')

if __name__ == "__main__":
    main()
