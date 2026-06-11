from jaad_data import JAAD
from pie_data import PIE
pie_path = '/data/chl343/PIE'
imdb = PIE(data_path=pie_path)
imdb.extract_and_save_images()