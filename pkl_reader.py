import gzip
import pickle 

with open('mocap_output/mocap/00000_prediction_result.pkl', 'rb') as f:
    data = pickle.load(f)

print(data)