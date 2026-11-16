import json,random,torch
if not torch.cuda.is_available(): raise SystemExit('CUDA required')
r=random.Random(784162); results=[]
for count in (2,3,4,8,16,64,512,4096):
 indices=[sorted(r.sample(range(4096),count)) for _ in range(512)]
 rows=torch.full((512,4096),float('-inf'),device='cuda')
 rows.scatter_(1,torch.tensor(indices,device='cuda'),0.)
 sorted_indices=torch.sort(rows,dim=-1,descending=True).indices[:,0].tolist()
 different=[{'row':i,'expected_first':v[0],'torch_choice':x} for i,(v,x) in enumerate(zip(indices,sorted_indices)) if v[0]!=x]
 results.append({'ties':count,'rows':len(indices),'first_index_differences':len(different),'examples':different[:5]})
print(json.dumps({'torch':torch.__version__,'device':torch.cuda.get_device_name(0),'rows':results},sort_keys=True))
