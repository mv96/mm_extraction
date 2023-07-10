import os
import subprocess
from tqdm import tqdm
from joblib import Parallel, delayed
import glob


class pdfalto_on_pdfs:
    def __init__(self, pdfalto, n_jobs, main_folder):
        self.pdfalto = pdfalto
        self.n_jobs = n_jobs
        self.main_folder = main_folder

    def fit(self):
        pdfs = glob.glob(os.path.join(self.main_folder, "*", "*.pdf"))
        print(len(pdfs))
        """
        print("generating pdf list")
        for folder in tqdm(self.sub_folders):
            try:
                all_pdfs=[os.path.join(self.main_folder,folder,element) 
                for element in os.listdir(os.path.join(self.main_folder,folder)) if element.endswith(".pdf")]
            except:
                continue
            if(len(all_pdfs)>=1):
                pdfs.append(all_pdfs[0])
                print(folder)
        print(len(pdfs))
        """

        """            
        for pdf in tqdm(pdfs):
            command_to_run=["{}".format(self.pdfalto),"-annotation","-readingOrder", "{}".format(pdf)]
            print("running pdfalto")
            subprocess.run(command_to_run)
        """

        Parallel(n_jobs=self.n_jobs)(
            delayed(subprocess.run)(
                [
                    "{}".format(self.pdfalto),
                    "-annotation",
                    "-readingOrder",
                    "{}".format(pdf),
                ]
            )
            for pdf in tqdm(pdfs)
        )

        return pdfs
