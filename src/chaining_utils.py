
import shutil
import os

# limpar todas as imagens em ../data/images
def chaining_remove_images():
    folder = '../data/images'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

# remove "../data/analise1.json", "../data/analise2.json", "../data/analise3.json" programatically
def chaining_remove_analises():
    for i in range(1, 4):
        try:
            os.remove(f"../data/analise{i}.json")
        except FileNotFoundError:
            pass