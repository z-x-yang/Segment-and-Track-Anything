import zipfile
import os


def zip_folder(source_folder, zip_dir):
    f = zipfile.ZipFile(zip_dir, 'w', zipfile.ZIP_DEFLATED)
    pre_len = len(os.path.dirname(source_folder))
    for dirpath, dirnames, filenames in os.walk(source_folder):
        for filename in filenames:
            pathfile = os.path.join(dirpath, filename)
            arcname = pathfile[pre_len:].strip(os.path.sep)
            f.write(pathfile, arcname)
    f.close()