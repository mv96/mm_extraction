from shutil import copy
from tqdm import tqdm
import os
import subprocess
from joblib import Parallel, delayed
from tqdm import tqdm


class tex2pdf:
    def __init__(
        self, source_directory, new_extthm_path=None, n_jobs=1, dest_pdfs=None
    ):
        self.source_directory = source_directory
        self.new_extthm_path = new_extthm_path
        self.n_jobs = n_jobs
        self.dest_pdfs = dest_pdfs

    def copy_extthm_file(self):
        """copies extthm file to the source subdirectories"""
        new_extthm_path = self.new_extthm_path

        def check_path_contains_styfile(folder):
            if "extthm.sty" in os.listdir(folder):
                return True
            else:
                return False

        valid_paths = []
        for element in os.listdir(self.source_directory):
            try:
                # this will fail for all the files that are not folders
                res = check_path_contains_styfile(
                    os.path.join(self.source_directory, element)
                )
                if res is True:
                    valid_paths.append(os.path.join(self.source_directory, element))
            except:
                print("error-- no style file found")
                continue

        for dst in tqdm(valid_paths):
            copy(new_extthm_path, dst)

        return valid_paths

    def find_main_tex_files(self, valid_paths):
        """finds the main tex file for
        each of the subdirectories in the source"""

        def read_tex_file_find_doc_class(texfile):
            found = False
            with open(texfile, encoding="utf8", errors="ignore") as f:
                lines = f.readlines()
                f.close()

            for line in lines:
                if line.lstrip().startswith("\\documentclass"):
                    found = True
                    break

            return found

        def check_tex_file_exists(folder):
            tex_files = [
                os.path.join(folder, element)
                for element in os.listdir(folder)
                if element.endswith(".tex")
            ]

            # basically the files that have a document class
            indexes_of_valid_files = []

            if len(tex_files) >= 1:
                states = []

                for texfile in tex_files:
                    state = read_tex_file_find_doc_class(
                        texfile
                    )  # look for document class
                    states.append(state)

                for i, element in enumerate(states):
                    if element is True:
                        indexes_of_valid_files.append(tex_files[i])

                if (
                    len(indexes_of_valid_files) == 1
                ):  # exactly on valid tex file with doc class
                    return indexes_of_valid_files[0]

                if len(indexes_of_valid_files) == 0:  # no valid document class
                    # file
                    # among all tex file #but then many tex files available
                    if len(tex_files) == 1:
                        return tex_files[0]
                    elif len(tex_files) >= 1:
                        # print("CASE A1- no doc class tex file and many tex avl")
                        return tex_files[0]
                    else:
                        # print("CASE A2 - no doc class tex file and no tex avl")
                        return None

                if len(indexes_of_valid_files) > 1:  # many valid doc class files
                    # print("CASE B- many doc class files available")
                    return indexes_of_valid_files[0]

            else:
                return tex_files[0]

        #########################main part####################

        tex_file_paths = []

        for path in valid_paths:
            # print(path)
            tex = check_tex_file_exists(path)
            if tex is not None:
                tex_file_paths.append(tex)

        return tex_file_paths

    def generate_pdf(self, tex_file):
        """generates pdf for the given tex file using pdflatex"""

        def check_if_output_is_valid(result):
            try:
                decoded_text = result.stdout.decode("utf-8").split("\n")[-20:]
            except:
                decoded_text = str(result).split("\n")[-20:]

            if len(decoded_text) == 1:
                decoded_text = decoded_text[0]

                # entire block is a string
                if "Output written on" in decoded_text:
                    return True
            else:
                for element in decoded_text:
                    if element.startswith("Output written on"):
                        return True
                    else:
                        continue
            return False

        ##latexmk -e '$pdflatex="pdflatex %O -interaction=nonstopmode %S"' -pdf <main_tex>
        cmd_1 = "latexmk -f -e"
        cmd_2 = str('$pdflatex="pdflatex %O -interaction=nonstopmode %S"')
        cmd_3 = "-pdf"
        cmd_2 = "'{}'".format(cmd_2)
        cmd = cmd_1 + " " + cmd_2 + " " + cmd_3

        def try_new(tex_filee):
            full = cmd + " " + str(tex_filee)
            result1 = subprocess.run(full, capture_output=True, timeout=30, shell=True)
            return result1

        def try_a(tex_filee):
            # print("a")
            result1 = subprocess.run(
                ["pdflatex", tex_filee], capture_output=True, timeout=20
            )
            return result1

        def try_b(tex_filee):
            # print("b")
            result2 = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", tex_filee],
                capture_output=True,
                timeout=20,
            )
            return result2

        def try_c(tex_filee):
            # print("c")
            result3 = subprocess.run(
                ["pdflatex", tex_filee], text=True, capture_output=True, timeout=20
            )
            return result3

        def try_d(tex_filee):
            # print("d")
            result4 = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", tex_filee],
                text=True,
                capture_output=True,
                timeout=20,
            )
            return result4

        # print(tex_file)
        folder_path, tex_filee = tex_file.rsplit("/", 1)
        os.chdir(folder_path)

        # run pdflatex
        for _try in [try_new]:  # try_b,try_d,try_c,try_a
            try:
                result = _try(tex_file)
                # if this does not work out we can simply return the number

                text = str(result.stdout)
                error_line = "Errors, so I did not complete making targets"
                if error_line in text:
                    # write it to the log_file
                    file2 = open("/Users/mv96/Desktop/errors.txt", "a")
                    file2.write(str(tex_file) + "\n")
                    file2.close()

            except subprocess.TimeoutExpired:
                continue
            except Exception as error:
                print(error)
                continue

            if result.returncode == 0 or check_if_output_is_valid(result):
                return (0, tex_file)
            else:
                continue

        return (1, tex_file)

    def create_logs(self, res):
        failed = []
        success = []
        for element in res:
            if element is None:
                continue
            if element[0] == 0:
                success.append(element[1].replace(".tex", ".pdf"))
            else:
                failed.append(element)

        return success, failed

    def copy_to_destination_folder(self, success):
        try:
            os.mkdir(self.dest_pdfs)
        except:
            pass

        for pdf in tqdm(success):
            folder_name = os.path.join(self.dest_pdfs, pdf.rsplit("/", 2)[1])
            # print(folder_name)
            # make this folder
            try:
                os.mkdir(folder_name)

                # copy pdf inside this folder
                copy(pdf, folder_name)
            except:
                continue

    def fit(self):
        """This fit method takes
        1.The extthm file and adds a reference to in the tex file
        2.Generates annotated pdfs
        3.Reports the result of success/failure

        Returns:
            _type_: _description_
        """
        valid_paths = self.copy_extthm_file()
        tex_file_paths = self.find_main_tex_files(valid_paths)
        current = os.getcwd()

        # there could be over flow of resources so one can reduce the n_jobs parameter
        res = Parallel(n_jobs=self.n_jobs, verbose=10)(
            delayed(self.generate_pdf)(tex) for tex in tqdm(tex_file_paths[:])
        )

        os.chdir(current)

        success, failed = self.create_logs(res)

        if self.dest_pdfs is not None:
            self.copy_to_destination_folder(success)

        return success, failed
