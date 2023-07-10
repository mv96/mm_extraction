import os
from bs4 import BeautifulSoup as bs
from joblib import Parallel, delayed
from tqdm import tqdm


class pdf_is_valid:
    def __init__(self, destination, n_jobs):
        self.n_jobs = n_jobs
        self.destination = destination

    def check_all_files(self, pdf):
        # if pdf, it's tei.xml , its xml file is present in the directory
        files_to_check = [
            pdf,
            pdf.replace(".pdf", ".xml"),
            pdf.replace(".pdf", ".tei.xml"),
            pdf.replace(".pdf", "_annot.xml"),
        ]
        status = []
        for file in files_to_check:
            status.append(os.path.exists(file))
        return all(status)

    def read_file_xml_to_bs4(self, xml_file):
        """reads xml returns bs4 object"""
        with open(xml_file, "r") as file:
            # Read each line in the file, readlines() returns a list of lines
            content = file.readlines()
            # Combine the lines in the list into a string
            content = "".join(content)
            bs_content = bs(content, "xml")
            file.close()

        return bs_content

    def check_valid_annotation(self, pdf):
        """check the annotation is valid also removes invalid annots"""
        annot_path = pdf.replace(".pdf", "_annot.xml")
        bs_content = self.read_file_xml_to_bs4(annot_path)
        annotations = bs_content.find_all("ANNOTATION")
        names = []
        for annotation in annotations:
            try:
                name = annotation.DEST.text
            except:
                continue
            names.append(name)
        for name in names:
            if name.startswith("uri:"):
                return True, pdf
        return False, pdf

    def fit(self):
        # filter all pdf names in the destinations

        pdfs = []
        for element in os.listdir(self.destination):
            main = os.path.join(self.destination, element)
            try:
                files = os.listdir(main)
            except:
                continue

            if len(files) == 0:
                continue
            pdf = list(filter(lambda x: x.endswith(".pdf"), files))[0]
            pdfs.append(os.path.join(main, pdf))

        filtered_valid = list(filter(lambda x: self.check_all_files(x), pdfs))

        pos = []
        neg = []
        res = Parallel(n_jobs=self.n_jobs)(
            delayed(self.check_valid_annotation)(pdf) for pdf in tqdm(filtered_valid)
        )
        for element in res:
            if element[0] is True:
                pos.append(element[1])
            else:
                neg.append(element[1])
        return pos, neg
