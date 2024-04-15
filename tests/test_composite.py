import os
import time
from icecream import ic
from andesite.composite.composite import Assay, Collar, DatafileComposite, Survey


assay_path = os.path.join('data', 'parker', 'Assay.csv')
collar_path = os.path.join('data', 'parker', 'Collar.csv')
survey_path = os.path.join('data', 'parker', 'Survey.csv')
assay = Assay(assay_path)
ic(assay.get_metadata())
collar = Collar(collar_path)
ic(collar.get_metadata())
survey = Survey(survey_path)
ic(survey.get_metadata())
init_time = time.time()
df_composite = DatafileComposite(assay, collar, survey)
df_composite.composite(['Cu_pct', 'Au_ppm', 'SAMPLETYPE', 'SAMPLEID'], 2, output_filename='composites_lab_2')
end_time = time.time()
print(f'Time elapsed: {(end_time - init_time):.3f} seconds')