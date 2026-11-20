import json,sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import numpy as np
import onnxruntime as ort
from deploy.quantize_int8 import BoardCalibration,quantize_model
from deploy.pack_onnx import pack
from model.artifacts import artifact_info
from chessio.records import record_board
from chessio.lc0 import policy_index
import chess

source='ckpts/sonofmagnus-cnn6M-fp32.onnx'
data='data/distill341k_train.jsonl'
calibration=BoardCalibration(data,limit=128,classical=True)
options=ort.SessionOptions();options.intra_op_num_threads=4
reference=ort.InferenceSession(source,sess_options=options,providers=['CPUExecutionProvider'])
inputs=calibration.boards
baseline=[reference.run(None,{'board':x})[0][0] for x in inputs]
legal=[[policy_index(move,board) for move in board.legal_moves] for board in calibration.positions]
def quality(path):
 session=ort.InferenceSession(path,sess_options=options,providers=['CPUExecutionProvider']);kl=[];same=0
 for x,p,indices in zip(inputs,baseline,legal):
  q=session.run(None,{'board':x})[0][0];p=p[indices].astype(np.float64);q=q[indices].astype(np.float64)
  pl=p-p.max();pl-=np.log(np.exp(pl).sum());ql=q-q.max();ql-=np.log(np.exp(ql).sum())
  kl.append(float((np.exp(pl)*(pl-ql)).sum()));same+=p.argmax()==q.argmax()
 return {'positions':len(inputs),'mean_legal_policy_kl':float(np.mean(kl)),'same_argmax':int(same),'model':artifact_info(path)}
results={'scope':'Precision selection on training calibration only; no Elo or test score inferred','data':artifact_info(data),'calibration_seed':0,'calibration_positions':128,'baseline':quality('ckpts/sonofmagnus-cnn6M-int8.onnx'),'variants':{}}
for name,indices in [('early',list(range(1,9))),('late',list(range(13,21)))]:
 unpacked=f'ckpts/sonofmagnus-cnn6M-int8-{name}-unpacked.onnx';packed=f'ckpts/sonofmagnus-cnn6M-int8-{name}.onnx'
 quantize_model(source,unpacked,calibration_data=data,keep_input_and_heads=True,keep_convs=indices)
 packing=pack(unpacked,packed);results['variants'][name]=dict(quality(packed),protected_convolutions=indices,packing=packing)
 print(json.dumps({name:results['variants'][name]}),flush=True)
Path('out/cnn6m_mixed_conv_probe.json').write_text(json.dumps(results,indent=2)+'\n')
