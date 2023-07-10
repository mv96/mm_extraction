import os
import shutil
from tqdm import tqdm


class copy_dest:
    def __init__(self, source, destination):
        self.directory = source
        self.dest = destination

    def copy(self):
        directory = self.directory
        dest = self.dest

        valid_pdfs = []
        for element in os.listdir(directory):
            folder_path = os.path.join(directory, element)
            try:
                for file in os.listdir(folder_path):
                    if file.endswith(".pdf"):
                        if os.path.exists(
                            os.path.join(folder_path, file.replace(".pdf", ".tex"))
                        ):
                            valid_pdfs.append(os.path.join(folder_path, file))
            except:
                continue

        # directory created
        try:
            os.mkdir(dest)
        except:
            pass

        for element in [element.split("/")[-2] for element in valid_pdfs]:
            try:
                os.mkdir(os.path.join(dest, element))
            except:
                continue

        # start copying files
        for element in tqdm(valid_pdfs):
            suff = "/" + "/".join(element.rsplit("/", 2)[1:])
            final = dest + suff

            source = element
            shutil.copy(source, final)
