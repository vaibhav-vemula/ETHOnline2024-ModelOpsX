import multiprocessing
import os  

processes = ('main.py', 'listener.py')                                    

def execute(process):
    if process == 'main.py':
        os.system(f'streamlit run {process}')
    else:
        os.system(f'python3 {process}')

process_pool = multiprocessing.Pool(processes = 2)                                                        
process_pool.map(execute, processes)