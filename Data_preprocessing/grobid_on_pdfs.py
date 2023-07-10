import subprocess
import os
from grobid_client_python.grobid_client.grobid_client import GrobidClient
import time
from tqdm import tqdm
from joblib import Parallel, delayed


class grobid_on_pdfs:
    def __init__(self, grobid_directory, grobid_client, sub_folders, n_jobs):
        self.grobid_directory = grobid_directory
        self.grobid_client = grobid_client
        self.sub_folders = sub_folders
        self.n_jobs = n_jobs

    def fit(self):
        valid_paths = self.sub_folders.copy()

        # do not run grobid on generated
        for pdf_path in tqdm(self.sub_folders):
            try:
                tei_xml = list(
                    filter(lambda x: x.endswith(".tei.xml"), os.listdir(pdf_path))
                )
            except:
                continue
            if len(tei_xml) == 1:
                valid_paths.remove(pdf_path)

        # go to the path and run the server initializing command
        current_directory = os.getcwd()

        # change to grobid directory
        os.chdir(self.grobid_directory)

        # run initialize the grobid server
        p = subprocess.Popen(["./gradlew", "run"])

        # give a 10 secs default waiting time for the grobid server to finish init
        time.sleep(120)

        client = GrobidClient()  # specify the config.json file
        time.sleep(5)

        Parallel(n_jobs=self.n_jobs)(
            delayed(client.process)(
                "processFulltextDocument",
                folder,
                n=20,
                generateIDs=True,
                consolidate_citations=True,
                tei_coordinates=True,
                segment_sentences=True,
                include_raw_affiliations=True,
                include_raw_citations=True,
            )
            for folder in tqdm(valid_paths[:])
        )

        # kill the process
        p.kill()
        # change back to the original directory
        os.chdir(current_directory)
